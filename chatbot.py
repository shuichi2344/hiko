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
import ollama
from faster_whisper import WhisperModel

# Optional GPIO stop button
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("📝 GPIO not available - running without button support")

# ===== Configuration =====
PTT_BUTTON_PIN = int(os.getenv("PTT_BUTTON_PIN", "17"))

# Preferred capture settings (we’ll auto-fallback if device refuses)
PREF_SAMPLE_RATE = 16000
PREF_CHANNELS = 1

# VAD settings
FRAME_MS = 30
SILENCE_THRESHOLD = 120   # Base RMS
END_SILENCE_MS = 800
MIN_SPEECH_MS = 300
MAX_RECORDING_MS = 15000

# Models
WHISPER_MODEL = "medium.en"
LLM_MODEL = "gemma3:270m"

# Piper voice/model settings
PIPER_MODEL = os.path.expanduser("~/hiko/piper-voices/en_US-ryan-high.onnx")
PIPER_CONFIG = os.path.expanduser("~/hiko/piper-voices/en_US-ryan-high.onnx.json")
PIPER_SPEAKER = ""  # Optional for multi-speaker models
PIPER_LENGTH_SCALE = "0.91"  # Slightly faster speech (equivalent to TTS_SPEED=1.1)

# Conversation
AUTO_RESTART_DELAY = 1.0
WAKE_WORDS = ["hey computer", "okay computer", "hey assistant"]

# Persona / system prompt
SYSTEM_PROMPT = (
    "You are Hiko."
    "Always give very short replies (max 2 sentences). "
    "Use simple words. Use plain ASCII only. Do not use emojis, emoticons, unicode symbols, markdown, or bullet points."
)

# Temp files
TEMP_WAV = Path("/tmp/recording.wav")
PIPER_OUT_WAV = Path("/tmp/tts_out.wav")

# Optional: force specific PipeWire nodes (id or name)
MIC_TARGET = os.environ.get("MIC_TARGET")
SINK_TARGET = os.environ.get("SINK_TARGET")

# ===== I/O backend toggle =====
# Force direct ALSA I/O (bypass PipeWire). Good when PipeWire doesn't show the HAT.
FORCE_ALSA = os.getenv("FORCE_ALSA", "0") == "1"
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "hw:0,0")  # card,device seen in arecord -l / aplay -l

USE_DEFAULT_ROUTING = os.getenv("DEFAULT_PIPEWIRE", "1") == "1"

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
        subprocess.run(["piper", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
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
    """
    Start a capture process that writes raw s16 PCM to stdout.

    - If FORCE_ALSA=1: use ALSA (arecord) on ALSA_DEVICE (e.g., hw:0,0).
    - Else: use PipeWire (pw-cat). When USE_DEFAULT_ROUTING is True,
      do NOT pass --target so it uses the default source like the reference.
      When USE_DEFAULT_ROUTING is False and `target` is provided,
      pass --target <id-or-name>.
    """
    if FORCE_ALSA:
        cmd = [
            "arecord",
            "-D", ALSA_DEVICE,
            "-f", "S16_LE",
            "-r", str(rate),
            "-c", str(channels),
            "-t", "raw"
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        cmd = [
            "pw-cat", "--record", "-",
            "--format", "s16",
            "--rate", str(rate),
            "--channels", str(channels),
        ]
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
        noise_samples = []
        if first_chunk:
            s = np.frombuffer(first_chunk, dtype=np.int16).astype(np.float32)
            noise_samples.append(float(np.sqrt(np.mean(s * s))))
        for _ in range(9):
            chunk = proc.stdout.read(frame_bytes)
            if chunk:
                s = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                noise_samples.append(float(np.sqrt(np.mean(s * s))))
        noise_floor = float(np.median(noise_samples)) if noise_samples else 50.0
        threshold = max(SILENCE_THRESHOLD, noise_floor * 1.8)
        print(f"   📏 Noise floor: {noise_floor:.1f}  |  Threshold: {threshold:.1f}")

        # ---- VAD state ----
        is_speaking = False
        silence_ms = 0
        speech_ms = 0
        total_ms = 0
        start = time.time()

        if first_chunk is not None:
            samples = np.frombuffer(first_chunk, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)))
            level = int(rms / 100)
            print(f"\r  Level: {'▁' * min(level, 20):<20} ", end="", flush=True)
            if rms > threshold:
                is_speaking = True
                speech_ms = FRAME_MS
                audio_buffer.extend(first_chunk)

        while True:
            if (time.time() - start) > timeout_seconds:
                if not is_speaking:
                    return None, None, None
                break

            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore").strip()
                if err:
                    print(f"\n❗ pw-cat: {err}")
                break

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)))
            level = int(rms / 100)
            print(f"\r  Level: {'▁' * min(level, 20):<20} ", end="", flush=True)

            if is_speaking:
                audio_buffer.extend(chunk)
                if rms < threshold:
                    silence_ms += FRAME_MS
                else:
                    silence_ms = 0
                    speech_ms += FRAME_MS

                if silence_ms >= END_SILENCE_MS and speech_ms >= MIN_SPEECH_MS:
                    dur_s = len(audio_buffer) / (rate * bytes_per_sample * ch)
                    print(f"\n  ✓ Recorded {dur_s:.1f}s")
                    break
                elif total_ms >= MAX_RECORDING_MS:
                    print("\n  ✓ Max recording length")
                    break
            else:
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
            try:
                proc.kill()
            except Exception:
                pass

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
                "temperature": 0.7,
                "num_predict": 60,
                "top_p": 0.9
            }
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "Hiko is having an issue right now."

# ---- Piper TTS ----
def _run_piper_to_wav(text: str, out_wav: Path) -> bool:
    """
    Run Piper CLI to synthesize `text` into `out_wav`.
    Returns True on success; prints Piper stderr on failure.
    """
    cmd = ["piper", "--model", PIPER_MODEL, "--output_file", str(out_wav)]
    if PIPER_CONFIG:
        cmd += ["--config", PIPER_CONFIG]
    if PIPER_SPEAKER:
        cmd += ["--speaker", PIPER_SPEAKER]
    if PIPER_LENGTH_SCALE:
        cmd += ["--length_scale", PIPER_LENGTH_SCALE]

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
    ptt_button = init_ptt_button()

    print("\n" + "="*50)
    print("🤖 VOICE CHATBOT READY!")
    print("="*50)
    print("Setup:")
    print("  • Microphone: ReSpeaker (PipeWire source)" if not USE_DEFAULT_ROUTING else " • Microphone: PipeWire default source")
    print("  • Speaker:    ReSpeaker (PipeWire sink)" if not USE_DEFAULT_ROUTING else "  • Speaker:    PipeWire default sink")
    print(f"  • Stop: {'Press Ctrl+C'}")
    if MIC_TARGET and not USE_DEFAULT_ROUTING:
        print(f"  • Mic target:  {MIC_TARGET}")
    if SINK_TARGET and not USE_DEFAULT_ROUTING:
        print(f"  • Sink target: {SINK_TARGET}")
    if ptt_button:
        print("\nHold the button to speak. Release to send.\n")
    else:
        print("\nListening for speech...\n")

    while True:
        try:
            # Push-to-talk path
            if ptt_button:
                audio_data, rate, ch = record_while_pressed(ptt_button, max_seconds=15)
            else:
                audio_data, rate, ch = record_with_vad(timeout_seconds=30)

            if audio_data:
                save_wav(audio_data, TEMP_WAV, sample_rate=rate, channels=ch)
                user_text = transcribe_audio(whisper_model, TEMP_WAV)

                if user_text:
                    print(f"📝 You said: \"{user_text}\"")
                    if any(w in user_text.lower() for w in ["goodbye", "bye", "stop", "exit", "quit", "shut down", "turn off"]):
                        speak_text(tts_pipeline, "Goodbye!")
                        break

                    reply = generate_response(user_text)
                    print(f"🤖 Assistant: \"{reply}\"\n")
                    speak_text(tts_pipeline, reply)

                    print(f"⏳ Ready again in {AUTO_RESTART_DELAY}s...")
                    time.sleep(AUTO_RESTART_DELAY)
                    if ptt_button:
                        print("\nHold the ReSpeaker button to speak. Release to send.\n")
                    else:
                        print("\nListening for speech (no GPIO detected)...\n")

                else:
                    print("❓ No speech detected in the captured audio\n")
            else:
                print("💤 No speech detected, still listening...\n")
                time.sleep(0.5)

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
