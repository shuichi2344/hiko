#!/usr/bin/env python3
import os, sys, time, socket, serial, re

# Use the SAME device you tested with
PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

TOKEN_RE = re.compile(r"\b(REC[_ ]START|REC[_ ]STOP)\b")

def send_to_control(cmd: str, timeout=0.6) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            resp = s.recv(64).decode("utf-8","ignore").strip()
            return resp.startswith("OK")
    except Exception:
        return False

def main():
    last_evt = None
    # Mirror your working reader: line mode, timeout=1, NO writes, NO RTS pulses
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print(f"[touch-ro] listening on {PORT} @ {BAUD} (read-only)")
        while True:
            try:
                line = ser.readline().decode("utf-8","ignore").strip()
            except Exception as e:
                print(f"[touch-ro] read error: {e}", file=sys.stderr)
                time.sleep(0.5)
                continue

            if not line:
                continue

            m = TOKEN_RE.search(line)
            if not m:
                # optional: uncomment to see extra lines
                # print("[touch-ro] non-token:", line)
                continue

            evt = "REC_START" if "START" in m.group(1) else "REC_STOP"
            if evt == last_evt:
                continue

            ok = send_to_control(evt)
            if ok:
                last_evt = evt
                print(f"[touch-ro] -> {evt}")
            else:
                print(f"[touch-ro] send failed: {evt}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
