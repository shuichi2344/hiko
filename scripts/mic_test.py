#!/usr/bin/env python3
"""
Mic test for ReSpeaker 2-Mic HAT using pvrecorder.
- Lists input devices
- Records N seconds at 16 kHz mono
- Saves to WAV
- Plays back with `aplay` (fallback to sounddevice if available)

Usage:
  python scripts/mic_test.py --seconds 5 --device 0 --outfile test.wav
"""

import argparse
import shutil
import subprocess
from array import array
from pvrecorder import PvRecorder
import wave

def list_devices():
    try:
        from pvrecorder import PvRecorder
        devices = PvRecorder.get_available_devices()
        print("\n=== Input devices ===")
        for i, d in enumerate(devices):
            print(f"[{i}] {d}")
        print("=====================\n")
    except Exception as e:
        print(f"Could not list devices: {e}")

def record(seconds: int, device_index: int, outfile: str, sample_rate: int = 16000, frame_length: int = 512):
    print(f"Recording {seconds}s @ {sample_rate} Hz, device_index={device_index} -> {outfile}")
    rec = PvRecorder(device_index=device_index, frame_length=frame_length)
    pcm = array('h')  # 16-bit signed

    try:
        rec.start()
        frames_needed = int(seconds * sample_rate / frame_length)
        for _ in range(frames_needed):
            frame = rec.read()     # list[int16] of length = frame_length
            pcm.extend(frame)
    finally:
        rec.stop()
        rec.delete()

    # write WAV: mono, 16-bit, sample_rate
    with wave.open(outfile, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    print(f"Saved: {outfile}")

def playback(outfile: str):
    print("Playing back...")
    # Prefer aplay if present
    if shutil.which("aplay"):
        subprocess.run(["aplay", outfile], check=False)
        return
    # Fallback: try sounddevice (optional)
    try:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(outfile, dtype='int16', always_2d=False)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"Playback fallback failed (install 'soundfile' to enable python playback): {e}")
        print("You can play manually with: aplay", outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--device", type=int, default=0, help="Device index from PvRecorder.get_available_devices()")
    parser.add_argument("--outfile", type=str, default="test.wav", help="Output WAV filename")
    args = parser.parse_args()

    list_devices()
    record(args.seconds, args.device, args.outfile)
    playback(args.outfile)

if __name__ == "__main__":
    main()
