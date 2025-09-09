#!/usr/bin/env python3
"""
Hiko demo voice loop:
- Record audio from mic (ReSpeaker) at 16 kHz mono
- Transcribe with whisper.cpp
- Send transcript to Ollama
- Speak the reply with Piper
- Play back audio with aplay
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from array import array
from pvrecorder import PvRecorder
import wave

# ---- CONFIG ----
WHISPER_BIN   = os.path.join("whisper.cpp", "build", "bin", "whisper-cli")
WHISPER_MODEL = os.path.join("whisper.cpp", "models", "ggml-small.en-q5_1.bin")

PIPER_MODEL   = os.path.join("piper-voices", "en_US-ryan-high.onnx")
OLLAMA_MODEL  = "llama3.2:1b"
# ----------------

def record_wav(seconds: int, device_index: int, outfile: str, sample_rate: int = 16000, frame_length: int = 512):
    print(f"[rec] {seconds}s @ {sample_rate} Hz from device_index={device_index} -> {outfile}")
    rec = PvRecorder(device_index=device_index, frame_length=frame_length)
    pcm = array('h')
    try:
        rec.start()
        frames_needed = int(seconds * sample_rate / frame_length)
        for _ in range(frames_needed):
            pcm.extend(rec.read())
    finally:
        rec.stop()
        rec.delete()

    with wave.open(outfile, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

def transcribe_with_whisper(wav_path: str) -> str:
    # Build the whisper.cpp command with improved decoding params
    cmd = (
        f"{shlex.quote(WHISPER_BIN)} "
        f"-t 4 "                                      # CPU threads
        f"-m {shlex.quote(WHISPER_MODEL)} "           # model path
        f"-f {shlex.quote(wav_path)} "                # input wav
        f"-l en "                                     # force English
        f"-bs 5 -bo 5 "                               # beam search settings
        f"-nt "                                       # no timestamps in output
        f"-otxt -of out/transcript"                   # write out/transcript.txt
    )

    subprocess.run(cmd, shell=True, check=True)

    txt_path = "out/transcript.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def chat_ollama(prompt: str) -> str:
    cmd = ["ollama", "run", OLLAMA_MODEL]
    try:
        res = subprocess.run(cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return res.stdout.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError as e:
        print(f"[ollama] error: {e.stderr.decode('utf-8', errors='ignore')}")
        return "Sorry, I had trouble thinking just now."

def speak_with_piper(text: str, wav_out: str):
    if not os.path.exists(PIPER_MODEL):
        print("[piper] no voice model found.")
        return
    cmd = f'piper --model {shlex.quote(PIPER_MODEL)} --output_file {shlex.quote(wav_out)}'
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    p.communicate(input=text.encode("utf-8"))

def play_wav(wav_path: str, device: str = None):
    cmd = ["aplay"]
    if device:
        cmd.extend(["-D", device])
    cmd.append(wav_path)
    subprocess.run(cmd, check=False)

def main():
    os.makedirs("out", exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--play-device", type=str, default=None)
    args = ap.parse_args()

    print("\nHiko voice demo: press Enter to record, Ctrl+C to quit.\n")

    try:
        while True:
            input(">> Press Enter to speak...")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_in = f"out/hiko_{ts}.wav"
            wav_reply = f"out/hiko_{ts}_reply.wav"

            record_wav(args.seconds, args.device, wav_in)
            transcript = transcribe_with_whisper(wav_in)
            if not transcript:
                print("[warn] empty transcript; skipping.")
                continue

            print(f"[you] {transcript}")
            prompt = f"You are Hiko, a cheerful, concise assistant for quick voice replies. Do NOT roleplay as an animal. No asterisks, no sound effects, no emojis. Be friendly and practical. Answer in 2–3 short sentences. The user said: {transcript}\nRespond briefly."
            reply = chat_ollama(prompt)
            print(f"[hiko] {reply}")

            speak_with_piper(reply, wav_reply)
            if os.path.exists(wav_reply):
                play_wav(wav_reply, args.play_device)

    except KeyboardInterrupt:
        print("\n[exit] Bye!")

if __name__ == "__main__":
    main()
