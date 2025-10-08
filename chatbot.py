#!/usr/bin/env python3
"""
Voice Chatbot (ReSpeaker Mic + ReSpeaker Speaker) — Piper TTS + PipeWire

Changes for ReSpeaker:
- Autodetect ReSpeaker input *and* output from `wpctl status` (no Bluetooth).
- New SINK_TARGET env/flag to force a specific PipeWire sink.
- pw-cat playback explicitly targets ReSpeaker sink when found.

TTS:
- Switched from Kokoro to Piper CLI.
- Synthesize to /tmp/tts_out.wav, then play via pw-cat/aplay.
"""

import sys
import os
import re
import signal
import time
import subprocess
import wave
import numpy as np
from pathlib import Path
import json
import random
import ollama
import threading
import socket
from faster_whisper import WhisperModel

try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("📝 GPIO not available - running without button support")

record_flag = threading.Event()

def start_recording():
    record_flag.set()

def stop_recording():
    record_flag.clear()

CONTROL_SOCK = "/tmp/hiko_control.sock"
#INVERT_TAP_SEQ = os.getenv("INVERT_TAP_SEQ", "0") == "1"

def _control_server():
    try:
        if os.path.exists(CONTROL_SOCK):
            os.remove(CONTROL_SOCK)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(CONTROL_SOCK)
        os.chmod(CONTROL_SOCK, 0o666)
        srv.listen(1)
        print(f"🧩 Control socket ready at {CONTROL_SOCK}")

        last_evt_ts = 0.0
        DEBOUNCE_MS = 120  # adjust if needed

        while True:
            conn, _ = srv.accept()
            with conn:
                data = conn.recv(64)
                if not data:
                    continue
                cmd = data.decode("utf-8", "ignore").strip().upper()

                # simple debounce
                now = time.monotonic() * 1000
                if (now - last_evt_ts) < DEBOUNCE_MS:
                    conn.sendall(b"OK\n")
                    continue
                last_evt_ts = now
                
                # - treat REC_STOP (release, LED ON) as "start recording" if idle
                # - treat REC_START (press, LED OFF) as "stop recording" if recording
                if cmd == "REC_START":  # press
                    if not record_flag.is_set():
                        print("🔔 REC_START (press) → START")
                        start_recording()
                    conn.sendall(b"OK\n")
                elif cmd == "REC_STOP":  # release
                    if record_flag.is_set():
                        print("🔕 REC_STOP (release) → STOP")
                        stop_recording()
                    conn.sendall(b"OK\n")
                else:
                    conn.sendall(b"ERR\n")
    except Exception as e:
        print(f"⚠️ control server error: {e}")

# ===== Configuration =====
PTT_BUTTON_PIN = int(os.getenv("PTT_BUTTON_PIN", "17"))
EXIT_ON_GOODBYE = os.getenv("EXIT_ON_GOODBYE", "0") == "1"  # default: do NOT exit

# Preferred capture settings (we’ll auto-fallback if device refuses)
PREF_SAMPLE_RATE = 16000
PREF_CHANNELS = 1

# VAD settings
FRAME_MS = 30
SILENCE_THRESHOLD = 120   # Base RMS
END_SILENCE_MS = 800
MIN_SPEECH_MS = 300
MAX_RECORDING_MS = 15000

# Extra VAD clamps
MIN_NOISE = 300.0      # clamp low
MAX_NOISE = 3000.0     # clamp high
TRIG_MULT = 3.0        # threshold multiplier over noise
MAX_TRIG  = 15000.0    # absolute max threshold
HANG_MS   = 400        # grace period after dips below threshold

# Manual stop only (no auto-stop on silence)
VAD_AUTO_STOP = False

def _rms_zeromean_int16(buf: bytes) -> float:
    if not buf:
        return 0.0
    s = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    if s.size == 0:
        return 0.0
    s = s - np.mean(s)  # DC removal
    return float(np.sqrt(np.mean(s * s)))

# Models
WHISPER_MODEL = "small.en"
LLM_MODEL = "gemma3:1b"


# Piper voice/model settings
PIPER_MODEL = os.path.expanduser("~/hiko/piper-voices/en_US-ryan-high.onnx")
PIPER_CONFIG = os.path.expanduser("~/hiko/piper-voices/en_US-ryan-high.onnx.json")
PIPER_SPEAKER = ""  # Optional for multi-speaker models
PIPER_LENGTH_SCALE = "1.4"  # Normal speed (1.0 is default)
PIPER_BIN = os.getenv("PIPER_BIN", "piper")

# Conversation
AUTO_RESTART_DELAY = 1.0
WAKE_WORDS = ["hello", "hi"]

# Persona / system prompt
SYSTEM_PROMPT = (
    "Your name is Hiko. Your favourite colour is blue. "
    "Always give short replies (max 2 sentences). If unsure, say 'I'm not sure.'"
    "Use plain ASCII only. Do not use emojis, emoticons, unicode symbols, markdown, or bullet points."
)

# Temp files
TEMP_WAV = Path("/tmp/recording.wav")
PIPER_OUT_WAV = Path("/tmp/tts_out.wav")
ERROR_TTS_WAV = Path("/tmp/tts_error.wav")

# Media
MUSIC_DIR = Path(os.path.expanduser("~/hiko/music"))
MUSIC_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Optional per-track copyright notices.
# - Keys should be lowercase filename stems (without extension)
# - Values are the copyright owner to display
# Example:
#   {
#       "river flows in you": "Yiruma",
#       "hallelujah": "Leonard Cohen Estate",
#   }
MUSIC_COPYRIGHT = {
    "lofi": "ggg.",
}

# Music playback state
MUSIC_CURRENT_PROC = None  # type: ignore
MUSIC_LAST_LIST = []       # list[Path]
MUSIC_INDEX = -1

# Quiz settings
QUIZ_DIR = Path(os.path.expanduser("~/.hiko/quizzes"))
QUIZ_STATE = {
    "active": False,
    "questions": [],
    "current_index": 0,
    "score": 0,
    "topic": ""
}

# Optional: force specific PipeWire nodes (id or name)
MIC_TARGET = os.environ.get("MIC_TARGET")
SINK_TARGET = os.environ.get("SINK_TARGET")

# ===== I/O backend toggle =====
# Force direct ALSA I/O (bypass PipeWire). Good when PipeWire doesn't show the HAT.
FORCE_ALSA = os.getenv("FORCE_ALSA", "0") == "1"
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "hw:0,0")  # card,device seen in arecord -l / aplay -l
FORCE_ALSA_CAPTURE = os.getenv("FORCE_ALSA_CAPTURE", "1") == "1"
ALSA_CAPTURE_DEVICE = os.getenv("ALSA_CAPTURE_DEVICE", "hw:0,0")

USE_DEFAULT_ROUTING = os.getenv("DEFAULT_PIPEWIRE", "1") == "1"

# ===== Error speech guard =====
_IN_ERROR_TTS = False
ERROR_SPOKEN_TEXT = "Sorry i have trouble hearing you. Please repeat again"

# ===== Init =====
def init_models():
    print("🚀 Starting Voice Chatbot...")
    print("📦 Loading models (first run may take a moment)...")

    print("  Loading Whisper...")
    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root=str(Path.home() / ".cache" / "whisper")
    )

    # Piper presence check
    print("  Checking Piper CLI...")
    try:
        subprocess.run([PIPER_BIN, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("❌ Piper not found. Install it and ensure `piper` is on PATH.")
        sys.exit(1)

    if not PIPER_MODEL:
        print("❌ PIPER_MODEL is not set. Example:")
        print("   export PIPER_MODEL=/home/pi/piper-voices/en_US-ryan-high.onnx")
        sys.exit(1)

    print("  Checking Ollama...")
    try:
        ollama.list()
    except Exception:
        print("❌ Ollama not running! Start it with: sudo systemctl enable --now ollama")
        sys.exit(1)

    print("✅ All models loaded successfully!\n")
    # For Piper we don't need a Python object; return a tiny wrapper (None placeholder)
    return whisper, None

def init_ptt_button():
    if not GPIO_AVAILABLE:
        print("📝 GPIO not available - push-to-talk disabled")
        return None
    try:
        # ReSpeaker button is active-low; pull_up=True works well.
        btn = Button(PTT_BUTTON_PIN, pull_up=True, bounce_time=0.03, hold_time=0.0)
        print(f"🔘 Push-to-talk button ready on GPIO {PTT_BUTTON_PIN} (press & hold to talk)")
        return btn
    except Exception as e:
        print(f"⚠️  Could not init PTT button: {e}")
        return None

# ===== ReSpeaker detection =====
_RESPEAKER_HINTS = ("respeaker", "seeed", "wm8960", "ac108", "voicecard")

def _parse_wpctl_ids(text):
    """
    Returns two dicts: inputs{id->name}, outputs{id->name}
    """
    inputs, outputs = {}, {}
    cur = None
    for line in text.splitlines():
        if line.strip().startswith("Audio"):
            cur = None
        if "Sinks:" in line:
            cur = "sinks"
            continue
        if "Sources:" in line:
            cur = "sources"
            continue
        m = re.search(r"^\s*(?:\*?\s*)?(\d+)\.\s+(.+?)\s+\[", line)
        if m and cur in ("sinks","sources"):
            _id, name = m.group(1), m.group(2)
            if cur == "sinks":
                outputs[_id] = name
            else:
                inputs[_id] = name
    return inputs, outputs

def _best_match(d, hints=_RESPEAKER_HINTS):
    # d: {id: name}
    if not d:
        return None
    # 1) exact-ish match on hints
    for _id, name in d.items():
        n = name.lower()
        if any(h in n for h in hints):
            return _id
    # 2) fallback
    return next(iter(d.keys()))

def detect_respeaker_targets():
    if USE_DEFAULT_ROUTING:
        return None, None
    try:
        out = subprocess.check_output(["wpctl", "status"], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"⚠️  Could not run `wpctl status`: {e}")
        return None, None
    inputs, outputs = _parse_wpctl_ids(out)
    mic_id = _best_match(inputs)
    sink_id = _best_match(outputs)
    if mic_id and sink_id:
        print(f"🎯 Detected ReSpeaker-ish source #{mic_id}: {inputs[mic_id]}")
        print(f"🎯 Detected ReSpeaker-ish sink   #{sink_id}: {outputs[sink_id]}")
    else:
        print("⚠️  Could not confidently detect ReSpeaker nodes. Will use PipeWire defaults.")
    return mic_id, sink_id

# ===== Helpers =====
def _spawn_record_process(rate, channels, target):
    if FORCE_ALSA_CAPTURE:
        cmd = ["arecord", "-D", ALSA_CAPTURE_DEVICE,
               "-f", "S16_LE", "-r", str(rate), "-c", str(channels), "-t", "raw"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        cmd = ["pw-cat", "--record", "-", "--format", "s16",
               "--rate", str(rate), "--channels", str(channels)]
        if target and not USE_DEFAULT_ROUTING:
            cmd += ["--target", str(target)]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _select_record_pipeline(target):
    """
    Try a few (rate,channels) combos so we don't crash if the device
    refuses 16k mono. Returns (proc, rate, channels, first_chunk or None, err_text).
    """
    attempts = [
        (PREF_SAMPLE_RATE, PREF_CHANNELS),
        (PREF_SAMPLE_RATE, 2),
        (48000, PREF_CHANNELS),
        (48000, 2),
    ]
    for rate, ch in attempts:
        proc = _spawn_record_process(rate, ch, target)
        bytes_per_sample = 2
        frame_bytes = int(rate * FRAME_MS / 1000) * bytes_per_sample * ch
        chunk = proc.stdout.read(frame_bytes)
        if chunk:
            return proc, rate, ch, chunk, ""
        err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore")
        try:
            proc.terminate(); proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if err.strip():
            print(f"   ⚠️  pw-cat refused {rate}Hz/{ch}ch: {err.strip()}")
        else:
            print(f"   ⚠️  pw-cat produced no data at {rate}Hz/{ch}ch, retrying...")
    return None, None, None, None, "No working pw-cat configuration found"

def record_with_vad(timeout_seconds=30):
    """Record audio until silence is detected (VAD)."""
    print("🎤 Listening... (speak now)")

    # Wait for tap
    if not record_flag.is_set():
        t0 = time.time()
        while not record_flag.is_set() and (time.time() - t0) < timeout_seconds:
            time.sleep(0.05)
        if not record_flag.is_set():
            return None, None, None

    effective_target = MIC_TARGET if (MIC_TARGET and not USE_DEFAULT_ROUTING) else None
    if effective_target:
        print(f"   🎯 Using source target: {effective_target}")

    proc, rate, ch, first_chunk, err = _select_record_pipeline(effective_target)
    if not proc:
        print(f"❌ {err}")
        return None, None, None

    bytes_per_sample = 2
    frame_bytes = int(rate * FRAME_MS / 1000) * bytes_per_sample * ch
    audio_buffer = bytearray()

    try:
        # ---- noise calibration (~300ms) ----
        noise_rms_vals = []
        if first_chunk:
            noise_rms_vals.append(_rms_zeromean_int16(first_chunk))
        for _ in range(9):  # ~9 * 30ms ≈ 270ms
            chunk = proc.stdout.read(frame_bytes)
            if chunk:
                noise_rms_vals.append(_rms_zeromean_int16(chunk))

        noise_floor = float(np.median(noise_rms_vals)) if noise_rms_vals else 50.0
        noise_floor = max(MIN_NOISE, min(noise_floor, MAX_NOISE))
        threshold = min(MAX_TRIG, max(SILENCE_THRESHOLD, noise_floor * TRIG_MULT))
        print(f"   📏 Calibrated noise RMS: {noise_floor:.1f} | Threshold: {threshold:.1f}")

        # ---- VAD state ----
        is_speaking, silence_ms, speech_ms, total_ms = False, 0, 0, 0
        start = time.time()

        if first_chunk is not None:
            rms = _rms_zeromean_int16(first_chunk)
            level = int(rms / 100)
            print(f"\r  Level: {'▁' * min(level, 20):<20} ", end="", flush=True)
            if rms > threshold:
                is_speaking = True
                speech_ms = FRAME_MS
                audio_buffer.extend(first_chunk)

        while True:
            # hard stop by tap
            if not record_flag.is_set():
                if len(audio_buffer) > 0:
                    dur_s = len(audio_buffer) / (rate * bytes_per_sample * ch)
                    print(f"\n  ⛔ Stopped by tap, keeping {dur_s:.1f}s")
                    break
                print("\n  ⛔ Stopped by tap, discarding (no audio)")
                _say_hearing_error()
                return None, None, None

            if (time.time() - start) > timeout_seconds:
                if not is_speaking:
                    return None, None, None
                if VAD_AUTO_STOP:
                    break
                else:
                    continue

            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore").strip()
                if err:
                    print(f"\n❗ pw-cat: {err}")
                break

            rms = _rms_zeromean_int16(chunk)
            level = int(rms / 100)
            print(f"\r  Level: {'▁' * min(level, 20):<20} ", end="", flush=True)

            if is_speaking:
                audio_buffer.extend(chunk)

                if VAD_AUTO_STOP:
                    # current behaviour (stop after HANG_MS of silence)
                    if rms < threshold:
                        silence_ms += FRAME_MS
                    else:
                        silence_ms = 0
                        speech_ms += FRAME_MS

                    if silence_ms >= HANG_MS and speech_ms >= MIN_SPEECH_MS:
                        dur_s = len(audio_buffer) / (rate * bytes_per_sample * ch)
                        print(f"\n  ✓ Recorded {dur_s:.1f}s")
                        break
                else:
                    # manual-stop mode: never end on silence, only by tap or safety cap
                    if total_ms >= MAX_RECORDING_MS:
                        print("\n  ⏱️ Max recording length")
                        break
            else:
                # wait for first speech to start buffering
                if rms > threshold:
                    is_speaking = True
                    speech_ms = FRAME_MS
                    silence_ms = 0
                    audio_buffer.extend(chunk)
                    print("\n  💬 Speech detected!")

            total_ms += FRAME_MS

    except KeyboardInterrupt:
        print("\n  ⏹️  Recording stopped")
        audio_buffer = None
    finally:
        try:
            proc.terminate(); proc.wait(timeout=0.8)
        except Exception:
            try: proc.kill()
            except Exception: pass

    if audio_buffer and len(audio_buffer) > 1000:
        return bytes(audio_buffer), rate, ch
    return None, None, None


def record_while_pressed(ptt_button, max_seconds=15, target=None):
    """
    Record raw PCM ONLY while ptt_button.is_pressed.
    """
    if not ptt_button:
        print("⚠️  No PTT button available; falling back to VAD.")
        return record_with_vad(timeout_seconds=max_seconds)

    print("🎤 Hold the button to talk...")

    try:
        while not ptt_button.is_pressed:
            time.sleep(0.01)
    except KeyboardInterrupt:
        return None, None, None

    effective_target = MIC_TARGET if (MIC_TARGET and not USE_DEFAULT_ROUTING) else None
    if effective_target:
        print(f"   🎯 Using source target: {effective_target}")

    proc, rate, ch, first_chunk, err = _select_record_pipeline(effective_target)
    if not proc:
        print(f"❌ {err}")
        return None, None, None

    bytes_per_sample = 2
    frame_bytes = int(rate * FRAME_MS / 1000) * bytes_per_sample * ch
    audio_buffer = bytearray()
    start = time.time()

    if first_chunk:
        audio_buffer.extend(first_chunk)

    print("  🎙️ Recording (release button to stop)")
    try:
        while True:
            if not ptt_button.is_pressed:
                break
            if (time.time() - start) >= max_seconds:
                print("  ⏱️  Max PTT length reached")
                break

            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore").strip()
                if err:
                    print(f"❗ pw-cat: {err}")
                break
            audio_buffer.extend(chunk)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.terminate(); proc.wait(timeout=0.8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if audio_buffer and len(audio_buffer) > 1000:
        dur_s = len(audio_buffer) / (rate * bytes_per_sample * ch)
        print(f"  ✓ Recorded {dur_s:.1f}s (PTT)")
        return bytes(audio_buffer), rate, ch

    print("  💤 Nothing captured (button released too fast?)")
    return None, None, None

def save_wav(audio_data, filepath, sample_rate, channels):
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

def transcribe_audio(whisper_model, audio_path):
    print("🧠 Transcribing...")
    try:
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="en",
            beam_size=3,
            patience=0.2,
            temperature=0.0,
            condition_on_previous_text=True,
            without_timestamps=True,
            initial_prompt=(
                "Transcribe short English voice commands with clear punctuation. "
                "Avoid filler words like um or uh."
            ),
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=450,
                speech_pad_ms=250
            ),
        )

        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list).strip()
        avg_logprob = (sum(getattr(s, "avg_logprob", -2.0) for s in seg_list) / len(seg_list)) if seg_list else -2.0

        if (len(text) < 3) or (avg_logprob < -1.0):
            segments2, _ = whisper_model.transcribe(
                str(audio_path),
                language="en",
                beam_size=5,
                patience=0.4,
                temperature=0.0,
                condition_on_previous_text=True,
                without_timestamps=True,
                initial_prompt=(
                    "Transcribe short English voice commands with clear punctuation. "
                    "Avoid filler words like um or uh."
                ),
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=450,
                    speech_pad_ms=250
                ),
            )
            seg_list2 = list(segments2)
            text2 = " ".join(s.text.strip() for s in seg_list2).strip()
            if text2:
                text = text2

        return text if text else None

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

def generate_response(user_text):
    print("💭 Thinking...")
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            options={
                "temperature": 0.6,
                "num_predict": 100,
                "top_p": 0.9
            }
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "Hiko is having an issue right now."

# ===== Light intent handling (task-specific helpers) =====


_JOKES = [
    "Why did the computer get cold? It forgot to close Windows.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why do Java developers wear glasses? Because they don't C#.",
    "I told my Pi a joke. It did not have enough GPIO pins to laugh.",
    "I tried to catch fog yesterday. I mist.",
    "I am reading a book on anti-gravity. It is impossible to put down.",
    "Why did the scarecrow win an award? He was outstanding in his field.",
    "I told my friend 10 jokes to make him laugh. Sadly, no pun in ten did.",
    "What do you call fake spaghetti? An impasta.",
    "I used to play piano by ear, now I use my hands.",
    "Why did the math book look sad? Too many problems.",
    "What do you call cheese that is not yours? Nacho cheese.",
    "Parallel lines have so much in common. It is a shame they will never meet.",
    "I ordered a chicken and an egg online. I will let you know which comes first.",
    "Why did the bicycle fall over? It was two tired.",
    "I asked a Frenchman if he played video games. He said Wii.",
    "I burnt 2000 calories today. I left my pizza in the oven.",
    "What do you call a belt made of watches? A waist of time.",
]

def classify_intent(text: str):
    t = (text or "").lower().strip()
    if any(k in t for k in ["joke", "make me laugh", "funny"]):
        return "joke"
    # Fixed command: only trigger when user starts with "play ..."
    if re.match(r"^play\b", t):
        return "music_play"
    if any(k in t for k in ["stop music", "stop the music", "music stop", "end music"]):
        return "music_stop"
    if any(k in t for k in ["next music", "next song", "skip song", "skip track", "next track"]):
        return "music_next"
    # Quiz commands
    if "question" in t and not any(k in t for k in ["stop", "end"]):
        return "quiz_start"
    if any(k in t for k in ["stop question", "end question", "quit question"]):
        return "quiz_stop"
    # If quiz is active, check for answer patterns
    # If quiz is active, only accept explicit True/False answers
    if QUIZ_STATE["active"]:
        if re.search(r"\b(true|false)\b", t):
            return "quiz_answer"

    return None

def _list_local_tracks():
    try:
        if not MUSIC_DIR.exists():
            return []
        return [p for p in MUSIC_DIR.iterdir() if p.suffix.lower() in MUSIC_EXTS]
    except Exception:
        return []

def _best_track_for_query(query: str, tracks):
    if not query:
        return None
    q = query.lower()
    # simple scoring: exact substring in filename wins; else token overlap
    best, best_score = None, 0
    q_tokens = [tok for tok in re.split(r"[^a-z0-9]+", q) if tok]
    for t in tracks:
        name = t.stem.lower()
        score = 0
        if q in name:
            score += 100
        else:
            for tok in q_tokens:
                if tok and tok in name:
                    score += 10
        if score > best_score:
            best, best_score = t, score
    return best

def _stop_music():
    global MUSIC_CURRENT_PROC
    try:
        if MUSIC_CURRENT_PROC and MUSIC_CURRENT_PROC.poll() is None:
            MUSIC_CURRENT_PROC.terminate()
            try:
                MUSIC_CURRENT_PROC.wait(timeout=1.0)
            except Exception:
                MUSIC_CURRENT_PROC.kill()
    except Exception:
        pass
    finally:
        MUSIC_CURRENT_PROC = None

def _play_track(path: Path):
    global MUSIC_CURRENT_PROC
    _stop_music()
    if FORCE_ALSA:
        MUSIC_CURRENT_PROC = subprocess.Popen(["aplay", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        cmd = ["pw-cat", "--playback", str(path)]
        if SINK_TARGET and not USE_DEFAULT_ROUTING:
            cmd += ["--target", str(SINK_TARGET)]
        MUSIC_CURRENT_PROC = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # After launching playback, announce copyright if configured
    try:
        stem = path.stem.lower()
        owner = MUSIC_COPYRIGHT.get(stem)
        if owner:
            notice = f"This track is copyrighted by {owner}."
            print(f"ℹ️  {notice}")
            # Speak the notice briefly without blocking the next cycle
            speak_text(None, notice)
    except Exception:
        pass

# ===== Quiz Functions =====
def _load_quiz(topic: str):
    """Load quiz questions from JSON file, accepting either:
       1) {"questions":[{"question","a","b","answer","explanation"}...]}
       2) {"items":[{"stem": {"en": ...}, "options":{"A","B"}, "answer":"A|B", "explanation":{"en": ...}}...]}
    """
    try:
        quiz_file = QUIZ_DIR / f"{topic}.json"
        if not quiz_file.exists():
            return None

        with open(quiz_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = data.get('questions')
        if not questions:
            items = data.get('items', [])
            # Normalize OpenTDB-style items -> expected shape
            questions = []
            for it in items:
                # question text
                qtext = it.get('question')
                if not qtext:
                    stem = it.get('stem', "")
                    if isinstance(stem, dict):
                        qtext = stem.get('en') or stem.get('EN') or ""
                    elif isinstance(stem, str):
                        qtext = stem
                # options
                opts = it.get('options', {})
                a_opt = (
                    it.get('a') or
                    opts.get('A') or opts.get('a') or
                    opts.get('True') or opts.get('true') or "True"
                )
                b_opt = (
                    it.get('b') or
                    opts.get('B') or opts.get('b') or
                    opts.get('False') or opts.get('false') or "False"
                )
                # explanation (allow dict or string)
                expl = it.get('explanation', "")
                if isinstance(expl, dict):
                    expl = expl.get('en') or expl.get('EN') or ""

                ans = (it.get('answer') or "").strip().upper()
                if ans not in ("A", "B"):
                    # Heuristic: if correct is True/False strings, map to A/B
                    if str(ans).lower() in ("true", "t"):
                        ans = "A"
                    elif str(ans).lower() in ("false", "f"):
                        ans = "B"
                    else:
                        ans = "A"  # safe default

                questions.append({
                    "question": qtext or "",
                    "a": a_opt,
                    "b": b_opt,
                    "answer": ans,
                    "explanation": expl or ""
                })

        # Final sanity filter: must have question + a + b + answer
        questions = [
            q for q in questions
            if q.get("question") and q.get("a") and q.get("b") and q.get("answer") in ("A", "B")
        ]

        random.shuffle(questions)
        return questions

    except Exception as e:
        print(f"Error loading quiz: {e}")
        return None


def _start_quiz(topic: str):
    """Start a new quiz session."""
    questions = _load_quiz(topic)
    if not questions:
        return f"Sorry, I couldn't find a {topic} quiz."
    
    QUIZ_STATE["active"] = True
    QUIZ_STATE["questions"] = questions[:3]  # Limit to 3 questions
    QUIZ_STATE["current_index"] = 0
    QUIZ_STATE["score"] = 0
    QUIZ_STATE["topic"] = topic
    
    return f"Starting {topic} quiz! I'll ask you {len(QUIZ_STATE['questions'])} questions. Please answer True or False."

def _ask_current_question():
    """Return the current question formatted for speech."""
    if not QUIZ_STATE["active"] or QUIZ_STATE["current_index"] >= len(QUIZ_STATE["questions"]):
        return None
    
    q = QUIZ_STATE["questions"][QUIZ_STATE["current_index"]]
    question_num = QUIZ_STATE["current_index"] + 1
    total = len(QUIZ_STATE["questions"])
    
    question_text = f"Question {question_num} of {total}. {q['question']} "
    question_text += "Is this statement True or False?"
    
    return question_text


def _check_answer(user_answer: str):
    """Check if the user's answer is correct and provide feedback.
       Accept only True/False. If invalid, re-ask the SAME question."""
    if not QUIZ_STATE["active"]:
        return "No quiz is active."

    current_q = QUIZ_STATE["questions"][QUIZ_STATE["current_index"]]
    correct_answer = current_q.get('answer', '').upper()

    # Normalize user answer (only True/False accepted)
    t = (user_answer or "").lower().strip()
    if t in ("true", "t"):
        user_choice = "A"
    elif t in ("false", "f"):
        user_choice = "B"
    else:
        # Do NOT advance; re-ask the SAME question
        reask = _ask_current_question() or "Please answer True or False."
        return "Sorry, I couldn't understand your answer. Please say True or False. " + reask

    is_correct = (user_choice == correct_answer)

    if is_correct:
        QUIZ_STATE["score"] += 1
        encouragement = random.choice([
            "Excellent! Well done!",
            "That's correct! Great job!",
            "Perfect! You're doing great!",
            "Right answer! Keep it up!"
        ])
    else:
        encouragement = random.choice([
            "Not quite right, but keep trying!",
            "That's not correct, but you're learning!",
            "Close! Don't give up!",
            "Not this time, but you'll get the next one!"
        ])

    explanation = current_q.get('explanation', '')
    tf = {"A": "True", "B": "False"}[correct_answer]
    feedback = f"{encouragement} The correct answer is {tf}."

    if explanation:
        feedback += f" {explanation}"

    # Now advance to the next question
    QUIZ_STATE["current_index"] += 1

    # If finished, close the quiz
    if QUIZ_STATE["current_index"] >= len(QUIZ_STATE["questions"]):
        final_score = QUIZ_STATE["score"]
        total_questions = len(QUIZ_STATE["questions"])
        percentage = int((final_score / total_questions) * 100)
        feedback += f" Quiz complete! You scored {final_score} out of {total_questions}, that's {percentage} percent!"
        QUIZ_STATE["active"] = False
        return feedback

    # Otherwise, append the next question
    next_q = _ask_current_question()
    if next_q:
        feedback += " " + next_q
    return feedback


def _stop_quiz():
    """Stop the current quiz."""
    if not QUIZ_STATE["active"]:
        return "No quiz is active."
    
    score = QUIZ_STATE["score"]
    answered = QUIZ_STATE["current_index"]
    
    QUIZ_STATE["active"] = False
    return f"Quiz stopped. You answered {score} out of {answered} questions correctly."

def handle_intent(intent: str, user_text: str):
    if intent == "joke":
        # Rotate through jokes for variety
        idx = int(time.time()) % len(_JOKES)
        return _JOKES[idx]
    if intent == "music_play":
        # Require explicit: "play <title>" or "play song <title>" / "play music <title>"
        t = (user_text or "").lower().strip()
        title = re.sub(r"^play(?:\s+(?:music|song|track))?\s*", "", t).strip()
        if not title:
            return "Say: play <title>. Example: play lofi"
        tracks = _list_local_tracks()
        if not tracks:
            return "No music files found. Add songs to your music folder."
        pick = _best_track_for_query(title, tracks) or tracks[int(time.time()) % len(tracks)]
        try:
            _play_track(pick)
            return f"Playing {pick.stem}."
        except Exception:
            return "Could not play music right now."
    if intent == "music_stop":
        _stop_music()
        return "Stopped."
    if intent == "music_next":
        tracks = _list_local_tracks()
        if not tracks:
            return "No music files found."
        pick = tracks[(int(time.time()) + 1) % len(tracks)]
        try:
            _play_track(pick)
            return f"Next: {pick.stem}."
        except Exception:
            return "Could not play next track."
    if intent == "quiz_start":
        t = (user_text or "").lower().strip()
        # strip trailing punctuation from STT (e.g., "science question.")
        t = re.sub(r"[^\w\s:-]", "", t)
        # Try pattern "<topic> question"
        topic = re.sub(r"\bquestions?\b.*$", "", t).strip()
        # If nothing left, try pattern "question: <topic>"
        if not topic:
            m = re.search(r"\bquestions?\b\s*[:\-]\s*(.+)$", t)
            topic = (m.group(1).strip() if m else "")
        # normalize to file-friendly name
        topic = topic.replace(" ", "_")
        if not topic:
            return "Say: <topic> question. Example: science question"
        response = _start_quiz(topic)
        if QUIZ_STATE["active"]:
            question = _ask_current_question()
            if question:
                response += " " + question
        return response
    if intent == "quiz_stop":
        return _stop_quiz()
    if intent == "quiz_answer":
    # _check_answer already appends the next question when appropriate
        return _check_answer(user_text)
    return None

# ---- Piper TTS ----
def _play_wav_file(out_path: Path):
    if FORCE_ALSA:
        play_cmd = ["aplay", "-D", ALSA_DEVICE, str(out_path)]
    else:
        play_cmd = ["pw-cat", "--playback", str(out_path)]
        if SINK_TARGET and not USE_DEFAULT_ROUTING:
            play_cmd += ["--target", str(SINK_TARGET)]
    return subprocess.Popen(play_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _say_hearing_error():
    """
    Speak a short fixed apology so the user knows to repeat.
    Guarded to avoid recursion; uses a separate WAV (ERROR_TTS_WAV).
    """
    global _IN_ERROR_TTS
    if _IN_ERROR_TTS:
        return
    _IN_ERROR_TTS = True
    try:
        synth_ok = ERROR_TTS_WAV.exists() and ERROR_TTS_WAV.stat().st_size > 44
        if not synth_ok:
            # Try to synthesize the error phrase
            if not _run_piper_to_wav(ERROR_SPOKEN_TEXT, ERROR_TTS_WAV):
                return  # If Piper itself fails, we can only log
        # Try to play it
        proc = _play_wav_file(ERROR_TTS_WAV)
        try:
            proc.communicate(timeout=10)
        except Exception:
            try: proc.kill()
            except Exception: pass
    except Exception:
        pass
    finally:
        _IN_ERROR_TTS = False

def _run_piper_to_wav(text: str, out_wav: Path) -> bool:
    """
    Run Piper CLI to synthesize `text` into `out_wav`.
    Returns True on success; prints Piper stderr on failure.
    """
    cmd = [PIPER_BIN, "-m", PIPER_MODEL, "-f", str(out_wav)]
    if PIPER_CONFIG:
        cmd += ["-c", PIPER_CONFIG]
    if PIPER_SPEAKER:
        cmd += ["-s", PIPER_SPEAKER]
    if PIPER_LENGTH_SCALE:
        cmd += ["--length-scale", PIPER_LENGTH_SCALE]

    # Piper expects newline-terminated lines from stdin
    payload = (text.strip() + "\n").encode("utf-8", errors="ignore")

    try:
        # Use run(..., input=...) so we don't manage pipes manually
        res = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            timeout=120
        )
        if res.returncode != 0:
            err = (res.stderr or b"").decode("utf-8", errors="ignore").strip()
            if err:
                print(f"❗ Piper error: {err}")
            else:
                print("❗ Piper failed with unknown error.")
            _say_hearing_error()
            return False
        # Sanity check the WAV actually exists and is non-trivial
        try:
            return out_wav.exists() and out_wav.stat().st_size > 44
        except Exception:
            return False
    except subprocess.TimeoutExpired:
        print("❗ Piper timed out")
        return False
    except Exception as e:
        print(f"❗ Piper failure: {e}")
        _say_hearing_error()
        return False

def speak_text(_unused_tts_pipeline, text):
    """
    Synthesize `text` with Piper (to /tmp/tts_out.wav) and play via:
      - ALSA (aplay) if FORCE_ALSA=1
      - PipeWire (pw-cat) otherwise
    When USE_DEFAULT_ROUTING is True, no --target is passed (default sink).
    When USE_DEFAULT_ROUTING is False and SINK_TARGET is set, --target is used.
    """
    print("🔊 Speaking...")
    try:
        # 1) Synthesize to WAV
        if PIPER_OUT_WAV.exists():
            try:
                PIPER_OUT_WAV.unlink()
            except Exception:
                pass

        if not _run_piper_to_wav(text, PIPER_OUT_WAV):
            print("❌ TTS Error: Piper synthesis failed")
            return

        # 2) Play it
        if FORCE_ALSA:
            play_cmd = ["aplay", "-D", ALSA_DEVICE, str(PIPER_OUT_WAV)]
        else:
            play_cmd = ["pw-cat", "--playback", str(PIPER_OUT_WAV)]
            if SINK_TARGET and not USE_DEFAULT_ROUTING:
                play_cmd += ["--target", str(SINK_TARGET)]

        proc = subprocess.Popen(play_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=120)
        if proc.returncode not in (0, None):
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()
            if err:
                print(f"❗ pw-cat/aplay playback: {err}")

    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        print("❗ Playback timed out")
    except Exception as e:
        print(f"❌ TTS Error: {e}")

def record_fixed_seconds(seconds=3):
    print(f"🎙️  Recording ~{seconds}s for test...")
    if MIC_TARGET:
        print(f"   🎯 Using source target: {MIC_TARGET}")

    proc, rate, ch, first_chunk, err = _select_record_pipeline(MIC_TARGET)
    if not proc:
        print(f"❌ {err}")
        return None, None, None

    bytes_per_sample = 2
    frame_bytes = int(rate * FRAME_MS / 1000) * bytes_per_sample * ch
    total_frames = int((seconds * 1000) / FRAME_MS)
    buf = bytearray()
    if first_chunk:
        buf.extend(first_chunk)

    try:
        for _ in range(total_frames - (1 if first_chunk else 0)):
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore").strip()
                if err:
                    print(f"❗ pw-cat: {err}")
                break
            buf.extend(chunk)
    finally:
        try:
            proc.terminate(); proc.wait(timeout=0.8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return (bytes(buf), rate, ch) if buf else (None, None, None)

# ===== Main =====
def main():
    global MIC_TARGET, SINK_TARGET
    args = sys.argv[1:]

    # Flags
    if "--mic-target" in args:
        try:
            MIC_TARGET = args[args.index("--mic-target") + 1]
        except Exception:
            print("⚠️  Usage: --mic-target <source-id-or-name>")
    if "--sink-target" in args:
        try:
            SINK_TARGET = args[args.index("--sink-target") + 1]
        except Exception:
            print("⚠️  Usage: --sink-target <sink-id-or-name>")

    if "--help" in args:
        print("Voice Chatbot - ReSpeaker mic+speaker via PipeWire (Piper TTS)")
        print("\nUsage: python3 chatbot.py [--mic-target <id-or-name>] [--sink-target <id-or-name>] [--test] [--list-audio]")
        sys.exit(0)

    if "--list-audio" in args:
        detect_respeaker_targets()
        sys.exit(0)

    # If user didn't force targets, try to auto-detect ReSpeaker
    if not MIC_TARGET or not SINK_TARGET:
        auto_mic, auto_sink = detect_respeaker_targets()
        MIC_TARGET = MIC_TARGET or auto_mic
        SINK_TARGET = SINK_TARGET or auto_sink

    # Quick test?
    if "--test" in args:
        data, rate, ch = record_fixed_seconds(seconds=3)
        if not data:
            print("❌ No audio captured during test.")
            sys.exit(1)

        # Downmix to mono if stereo
        if ch == 2:
            pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
            mono = ((pcm[:, 0].astype(np.int32) + pcm[:, 1].astype(np.int32)) // 2).astype(np.int16)
            data = mono.tobytes()
            ch = 1

        out = Path("/tmp/test.wav")
        save_wav(data, out, sample_rate=rate, channels=ch)

        print("▶️  Playing back test recording on ReSpeaker sink..." if (SINK_TARGET and not USE_DEFAULT_ROUTING) else "Playing back test recording...")
        if FORCE_ALSA:
            play_cmd = ["aplay", "-D", ALSA_DEVICE, str(out)]
        else:
            play_cmd = ["pw-cat", "--playback", str(out)]
            if SINK_TARGET and not USE_DEFAULT_ROUTING:
                play_cmd += ["--target", str(SINK_TARGET)]

        subprocess.run(play_cmd, check=False)
        print("✅ Audio test complete!")
        sys.exit(0)

    whisper_model, tts_pipeline = init_models()
    ptt_button =  None #init_ptt_button()

    # Start control socket thread
    threading.Thread(target=_control_server, daemon=True).start()

    print("\n" + "="*50)
    print("🤖 VOICE CHATBOT READY!")
    print("="*50)
    print("Setup:")
    print(f"  • Microphone: {'ALSA ' + ALSA_CAPTURE_DEVICE if FORCE_ALSA_CAPTURE else ('ReSpeaker (PipeWire source)' if not USE_DEFAULT_ROUTING else 'PipeWire default source')}")
    print(f"  • Speaker:    {'ALSA ' + ALSA_DEVICE if FORCE_ALSA else ('ReSpeaker (PipeWire sink)' if not USE_DEFAULT_ROUTING else 'PipeWire default sink')}")
    print(f"  • Stop: {'Press Ctrl+C'}")
    if MIC_TARGET and not USE_DEFAULT_ROUTING:
        print(f"  • Mic target:  {MIC_TARGET}")
    if SINK_TARGET and not USE_DEFAULT_ROUTING:
        print(f"  • Sink target: {SINK_TARGET}")

    print("\nTap the sensor to start. Tap again to stop.\n")

    while True:
        try:
            # 👉 Wait for REC_START from STM32 (touch_bridge.py calls start_recording())
            record_flag.wait()
            if not record_flag.is_set():
                continue  # got cleared before we started

            # One capture session (stoppable mid-way by REC_STOP)
            audio_data, rate, ch = record_with_vad(timeout_seconds=30)

            # We are done with this turn → go back to idle.
            # Clear the flag locally so we’re ready for the next REC_START.
            # (If the user taps a STOP after VAD ended, clearing again is harmless.)
            record_flag.clear()

            if not audio_data:
                print("💤 No speech captured (stopped or silence). Waiting for tap...\n")
                continue
            # If we captured stereo, downmix to mono for Whisper
            if ch == 2 and audio_data:
                pcm = np.frombuffer(audio_data, dtype=np.int16).reshape(-1, 2)
                mono = ((pcm[:, 0].astype(np.int32) + pcm[:, 1].astype(np.int32)) // 2).astype(np.int16)
                audio_data = mono.tobytes()
                ch = 1

            save_wav(audio_data, TEMP_WAV, sample_rate=rate, channels=ch)
            user_text = transcribe_audio(whisper_model, TEMP_WAV)
            print(f"   📝 Transcript debug: {repr(user_text)}")


            if user_text:
                print(f"📝 You said: \"{user_text}\"")
                if re.search(r"\b(goodbye|bye)\b", user_text.lower()):
                    speak_text(tts_pipeline, "Goodbye!")
                    if EXIT_ON_GOODBYE:
                        break
                    QUIZ_STATE["active"] = False
                    print("\nWaiting for tap...\n")
                    continue

                # Fast path: handle light intents locally
                intent = classify_intent(user_text)
                if QUIZ_STATE["active"] and intent is None:
                    reply = "Please answer True or False. " + (_ask_current_question() or "")
                else:
                    if intent:
                        reply = handle_intent(intent, user_text)
                    else:
                        reply = generate_response(user_text)

                print(f"🤖 Assistant: \"{reply}\"\n")
                speak_text(tts_pipeline, reply)

                print(f"⏳ Ready again in {AUTO_RESTART_DELAY}s...")
                time.sleep(AUTO_RESTART_DELAY)       
                print("\nWaiting for tap...\n")
            else:
                print("❓ No speech detected after transcription\n")
                _say_hearing_error()
                print("\nWaiting for tap...\n")

                """if ptt_button:
                    print("\nHold the ReSpeaker button to speak. Release to send.\n")
                else:
                    print("\nListening for speech (no GPIO detected)...\n")

            else:
                print("❓ No speech detected in the captured audio\n")
                _say_hearing_error()
        else:
            print("💤 No speech detected, still listening...\n")
            time.sleep(0.5)"""

        except KeyboardInterrupt:
            print("\n\n⌨️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Restarting in 3 seconds...\n")
            time.sleep(3)

    print("\n👋 Goodbye!")
    print("="*50)

if __name__ == "__main__":
    main()
