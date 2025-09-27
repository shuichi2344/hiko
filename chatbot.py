#!/usr/bin/env python3
"""
Voice Chatbot (ReSpeaker Mic + ReSpeaker Speaker) — TTS Tensor-safe + PipeWire

Changes for ReSpeaker:
- Autodetect ReSpeaker input *and* output from `wpctl status` (no Bluetooth).
- New SINK_TARGET env/flag to force a specific PipeWire sink.
- pw-cat playback explicitly targets ReSpeaker sink when found.
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
import hashlib
import ollama
from kokoro import KPipeline
from faster_whisper import WhisperModel

# Lock thread pools to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Pin PyTorch threads for Kokoro optimization
try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

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

# VAD settings - optimized for speed
FRAME_MS = 20  # Reduced from 30 for faster processing
SILENCE_THRESHOLD = 100   # Reduced from 120 for more sensitive detection
END_SILENCE_MS = 600  # Reduced from 800 for faster response
MIN_SPEECH_MS = 200  # Reduced from 300 for faster detection
MAX_RECORDING_MS = 12000  # Reduced from 15000 for faster processing

# Models
WHISPER_MODEL = "medium.en"
LLM_MODEL = "gemma3:270m"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.1

# Performance optimizations
WHISPER_BEAM_SIZE = 1  # Use greedy decoding for maximum speed
TTS_CHUNK_SIZE = 1024  # Optimized chunk size for streaming

# Response cache for common phrases
RESPONSE_CACHE = {}
CACHE_SIZE_LIMIT = 50  # Maximum number of cached responses

# TTS cache for instant repeated responses
TTS_CACHE_DIR = "/tmp/tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# Conversation
AUTO_RESTART_DELAY = 1.0
WAKE_WORDS = ["hey computer", "okay computer", "hey assistant"]

# Persona / system prompt
SYSTEM_PROMPT = (
    "You are Hiko."
    "Always give very short replies (max 2 sentences). "
    "Use simple words. Use plain ASCII only. Do not use emojis, emoticons, unicode symbols, markdown, or bullet points."
)


# Temp file
TEMP_WAV = Path("/tmp/recording.wav")

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
        download_root=str(Path.home() / ".cache" / "whisper"),
        local_files_only=False  # Allow model caching for faster subsequent loads
    )

    print("  Loading Kokoro TTS...")
    tts = KPipeline(lang_code='a')
    
    # Warm up TTS to prevent slow first utterance
    print("  Warming up TTS...")
    _ = list(tts("Hi", voice=TTS_VOICE, speed=TTS_SPEED))  # primes weights/kernels

    print("  Checking Ollama...")
    try:
        ollama.list()
    except Exception:
        print("❌ Ollama not running! Start it with: sudo systemctl enable --now ollama")
        sys.exit(1)

    print("✅ All models loaded successfully!\n")
    return whisper, tts

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
    # 2) fallback to the currently selected (*) if visible in wpctl status (marked elsewhere),
    #    but here we only have plain dict — so fallback to first as last resort.
    return next(iter(d.keys()))

def detect_respeaker_targets():
    if USE_DEFAULT_ROUTING:
        # behave like the reference: do not auto-detect/force nodes
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
        # arecord -> raw, 16-bit little-endian to stdout
        cmd = [
            "arecord",
            "-D", ALSA_DEVICE,
            "-f", "S16_LE",
            "-r", str(rate),
            "-c", str(channels),
            "-t", "raw"  # write to stdout
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # PipeWire path
        cmd = [
            "pw-cat", "--record", "-",
            "--format", "s16",
            "--rate", str(rate),
            "--channels", str(channels),
        ]
        # Only target a specific source when not using default routing
        if target and not USE_DEFAULT_ROUTING:
            cmd += ["--target", str(target)]

        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _select_record_pipeline(target):
    """
    Try a few (rate,channels) combos so we don't crash if the device
    refuses 16k mono. Returns (proc, rate, channels, first_chunk or None, err_text).
    """
    attempts = [
        (PREF_SAMPLE_RATE, PREF_CHANNELS),  # 16k / mono
        (PREF_SAMPLE_RATE, 2),              # 16k / stereo
        (48000, PREF_CHANNELS),             # 48k / mono
        (48000, 2),                         # 48k / stereo
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
    """Record audio until silence is detected (VAD).
    Returns (bytes, rate, channels) or (None, None, None).
    """
    print("🎤 Listening... (speak now)")

    # Only mention/force a source when not using default routing
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
        # ---- quick noise calibration (~300ms) ----
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

        # consider the first chunk
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
            # timeout if no speech
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
    Stops when button is released, or max_seconds elapse.
    Returns (bytes, rate, channels) or (None, None, None).
    """
    if not ptt_button:
        print("⚠️  No PTT button available; falling back to VAD.")
        return record_with_vad(timeout_seconds=max_seconds)

    print("🎤 Hold the button to talk...")

    # Wait for initial press (non-blocking check so Ctrl+C still works)
    try:
        while not ptt_button.is_pressed:
            time.sleep(0.01)
    except KeyboardInterrupt:
        return None, None, None

    # Only force a source when not using default routing
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

    # Seed with first chunk so users don't lose their first syllable
    if first_chunk:
        audio_buffer.extend(first_chunk)

    print("  🎙️ Recording (release button to stop)")
    try:
        while True:
            # Stop if released or max time exceeded
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

def _bytes_pcm16_to_float32_mono(b, channels, sr):
    """Convert raw PCM16 bytes to float32 mono array for zero-copy STT."""
    pcm = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)  # downmix
    # If your capture is not 16k, FW will resample internally; that's fine.
    return pcm

def transcribe_audio_array(whisper_model, audio_f32):
    """Transcribe audio directly from float32 array (zero-copy)."""
    print("🧠 Transcribing...")
    try:
        segments, info = whisper_model.transcribe(
            audio_f32,               # <-- array, not a file path
            language="en",
            beam_size=1,             # greedy: fastest; medium.en is strong enough
            temperature=0.0,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450, speech_pad_ms=250),
            condition_on_previous_text=True,
            initial_prompt="Transcribe short English voice commands with clear punctuation."
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text or None
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

def transcribe_audio(whisper_model, audio_path):
    print("🧠 Transcribing...")
    try:
        # ---- 1st pass: optimized for speed while maintaining accuracy ----
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="en",
            beam_size=WHISPER_BEAM_SIZE,  # Optimized beam size for speed
            temperature=0.0,
            condition_on_previous_text=False,  # Disabled for speed (minimal accuracy impact)
            without_timestamps=True,     # we don't need word times → small speed win
            initial_prompt=(
                "Transcribe short English voice commands with clear punctuation. "
                "Avoid filler words like um or uh."
            ),
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,  # Reduced for faster processing
                speech_pad_ms=150             # Reduced for faster processing
            ),
        )

        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list).strip()

        # Confidence proxy: average segment logprob (faster-whisper exposes this)
        avg_logprob = (sum(getattr(s, "avg_logprob", -2.0) for s in seg_list) / len(seg_list)) if seg_list else -2.0

        # ---- Smart fallback: only retry harder if first pass looks bad ----
        if (len(text) < 3) or (avg_logprob < -1.0):
            # Second pass with beam search for better accuracy when needed
            segments2, _ = whisper_model.transcribe(
                str(audio_path),
                language="en",
                beam_size=2,  # Small beam for fallback accuracy
                patience=0.1,  # Conservative patience value
                temperature=0.0,
                condition_on_previous_text=False,  # Disabled for speed
                without_timestamps=True,
                initial_prompt=(
                    "Transcribe short English voice commands with clear punctuation. "
                    "Avoid filler words like um or uh."
                ),
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,  # Reduced for speed
                    speech_pad_ms=150             # Reduced for speed
                ),
            )
            seg_list2 = list(segments2)
            text2 = " ".join(s.text.strip() for s in seg_list2).strip()
            if text2:
                text = text2  # use the improved result

        return text if text else None

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None


def generate_response(user_text):
    # Check cache first for common responses
    user_lower = user_text.lower().strip()
    if user_lower in RESPONSE_CACHE:
        print("💭 Using cached response...")
        return RESPONSE_CACHE[user_lower]
    
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
        response = resp["message"]["content"].strip()
        
        # Cache the response if it's short and common
        if len(response) < 100 and len(RESPONSE_CACHE) < CACHE_SIZE_LIMIT:
            RESPONSE_CACHE[user_lower] = response
        
        return response
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "Hiko is having an issue right now."

# ---- TTS utils (Tensor-safe) ----
def _to_numpy_audio(audio):
    """Convert various audio containers (torch.Tensor, list, np.ndarray) to 1-D float32 NumPy array."""
    try:
        import torch  # only for isinstance check
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
    except Exception:
        pass
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    return audio

def _tts_cache_path(text, voice, speed, sr):
    """Generate cache file path for TTS audio."""
    h = hashlib.sha1(f"{voice}|{speed}|{sr}|{text}".encode()).hexdigest()
    return os.path.join(TTS_CACHE_DIR, f"{h}.pcm")

def speak_text(tts_pipeline, text):
    """
    Synthesize `text` with Kokoro and play it out via:
      - ALSA (aplay) if FORCE_ALSA=1
      - PipeWire (pw-cat) otherwise
    When USE_DEFAULT_ROUTING is True, no --target is passed (default sink).
    When USE_DEFAULT_ROUTING is False and SINK_TARGET is set, --target is used.
    """
    print("🔊 Speaking...")
    try:
        # Use Kokoro's native sample rate to avoid pitch distortion
        sr = int(getattr(tts_pipeline, "sample_rate", 24000) or 24000)
        
        # Check TTS cache first
        cache_file = _tts_cache_path(text, TTS_VOICE, TTS_SPEED, sr)
        if os.path.exists(cache_file):
            print("🔊 Using cached TTS...")
            if FORCE_ALSA:
                play_cmd = ["aplay", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(sr), "-c", "1", cache_file]
            else:
                play_cmd = ["pw-cat", "--playback", cache_file, "--format", "s16", "--rate", str(sr), "--channels", "1"]
                if SINK_TARGET and not USE_DEFAULT_ROUTING:
                    play_cmd += ["--target", str(SINK_TARGET)]
            subprocess.run(play_cmd, check=False)
            return

        # Build playback command
        if FORCE_ALSA:
            play_cmd = [
                "aplay",
                "-D", ALSA_DEVICE,
                "-f", "S16_LE",
                "-r", str(sr),
                "-c", "1",
            ]
        else:
            play_cmd = [
                "pw-cat", "--playback", "-",
                "--format", "s16",
                "--rate", str(sr),
                "--channels", "1",
            ]
            # Only target a specific sink when not using default routing
            if SINK_TARGET and not USE_DEFAULT_ROUTING:
                play_cmd += ["--target", str(SINK_TARGET)]

        # Start a single playback process and stream PCM into it
        proc = subprocess.Popen(play_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Generate TTS audio and stream it with optimized chunking + caching
        gen = tts_pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED)
        audio_buffer = bytearray()
        
        # Write to cache file while streaming
        with open(cache_file, "wb") as cf:
            for _, _, audio in gen:
                # Convert to float32 NumPy and then to 16-bit PCM
                audio_np = _to_numpy_audio(audio)
                pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                audio_buffer.extend(pcm16)
                cf.write(pcm16)  # Write to cache
                
                # Stream in optimized chunks for better performance
                if len(audio_buffer) >= TTS_CHUNK_SIZE:
                    try:
                        if proc.stdin:
                            proc.stdin.write(audio_buffer)
                            audio_buffer.clear()
                    except BrokenPipeError:
                        # Playback process died; stop streaming further
                        break
            
            # Write remaining audio buffer
            if audio_buffer and proc.stdin:
                try:
                    proc.stdin.write(audio_buffer)
                except BrokenPipeError:
                    pass

        # Close stdin to signal EOF, then collect any errors
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass

        stderr = b""
        try:
            stderr = proc.stderr.read() if proc.stderr else b""
        except Exception:
            pass

        ret = None
        try:
            ret = proc.wait(timeout=5)
        except Exception:
            # If it hangs, try to terminate
            try:
                proc.terminate()
                ret = proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if ret not in (0, None):
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()
            if err:
                print(f"❗ pw-cat/aplay playback: {err}")

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
        print("Voice Chatbot - ReSpeaker mic+speaker via PipeWire")
        print("\nUsage: python3 chatbot.py [--mic-target <id-or-name>] [--sink-target <id-or-name>] [--test] [--list-audio]")
        sys.exit(0)

    if "--list-audio" in args:
        # show what we detect
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

            # Push-to-talk path: ONLY record while button is held.
            if ptt_button:
                audio_data, rate, ch = record_while_pressed(ptt_button, max_seconds=15)
            else:
                # Fallback to VAD if no GPIO available
                audio_data, rate, ch = record_with_vad(timeout_seconds=30)

            if audio_data:
                # Zero-copy STT: convert directly to float32 array
                audio_f32 = _bytes_pcm16_to_float32_mono(audio_data, channels=ch, sr=rate)
                user_text = transcribe_audio_array(whisper_model, audio_f32)
                
                # Optional: still save WAV for debugging if needed
                # save_wav(audio_data, TEMP_WAV, sample_rate=rate, channels=ch)

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