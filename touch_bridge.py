#!/usr/bin/env python3
import os, sys, time, socket, serial, re

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
TOKEN_RE = re.compile(r"\b(REC[_ ]START|REC[_ ]STOP)\b")

def send_to_control(cmd: str, timeout=0.8) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip()+"\n").encode())
            resp = s.recv(64).decode("utf-8","ignore").strip()
            return resp.startswith("OK")
    except Exception as e:
        print(f"[touch-ro] control send error: {e}")
        return False

def main():
    last = None
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print(f"[touch-ro] listening on {PORT} @ {BAUD} (read-only)")
        while True:
            line = ser.readline().decode("utf-8","ignore").strip()
            if not line:
                continue
            m = TOKEN_RE.search(line)
            if not m:
                continue
            evt = "REC_START" if "START" in m.group(1) else "REC_STOP"
            if evt == last:
                continue
            if send_to_control(evt):
                last = evt
                print(f"[touch-ro] -> {evt}")
            else:
                print(f"[touch-ro] send failed: {evt}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
