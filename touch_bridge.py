#!/usr/bin/env python3
# touch_bridge.py (robust)
import os, sys, time, socket, serial, glob, re

CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
BAUD = 115200
MIRROR_POSE_TO_STM32 = os.environ.get("HIKO_MIRROR_POSE", "1") == "1"

def _pick_port():
    # 1) explicit env
    envp = os.environ.get("HIKO_SERIAL_PORT")
    if envp and os.path.exists(envp):
        return envp
    # 2) by-id pattern
    matches = sorted(glob.glob("/dev/serial/by-id/usb-STMicroelectronics_*"))
    if matches:
        return matches[0]
    # 3) common ACM fallbacks
    for p in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"):
        if os.path.exists(p):
            return p
    return None

def _map_line_to_event(s: str):
    u = s.strip().upper()
    if re.search(r"\bREC[_ ]?START\b", u):
        return "REC_START"
    if re.search(r"\bREC[_ ]?STOP\b", u):
        return "REC_STOP"
    # tolerant to your older debug style
    if "RECORD_REQ=1" in u:
        return "REC_START"
    if "RECORD_REQ=0" in u:
        return "REC_STOP"
    return None

def _send_to_control(cmd: str):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd + "\n").encode("utf-8"))
            try: s.recv(16)
            except Exception: pass
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)

def main():
    last = None
    while True:
        port = _pick_port()
        if not port:
            print("[touch] no serial port yet; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last = None
            continue

        try:
            with serial.Serial(port, BAUD, timeout=0.2) as ser:
                print(f"[touch] listening on {port} @ {BAUD}")
                try: ser.reset_input_buffer()
                except Exception: pass

                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "ignore").strip()
                    evt = _map_line_to_event(line)
                    print(f"[touch] <- {line}")
                    if not evt:
                        continue
                    if evt == last:
                        continue
                    last = evt

                    _send_to_control(evt)
                    # optional: pose mirror
                    if MIRROR_POSE_TO_STM32:
                        try:
                            ser.write(b"POSE RECORD\n" if evt == "REC_START" else b"POSE IDLE\n")
                            ser.flush()
                        except Exception as e:
                            print(f"[touch] serial write error: {e}", file=sys.stderr)

        except Exception as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last = None

if __name__ == "__main__":
    main()
