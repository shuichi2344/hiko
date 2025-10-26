#!/usr/bin/env python3
import sys, time, glob
import serial  # pip install pyserial

# default: stable symlink if present, else /dev/ttyACM0
_DEFAULT_PORTS = sorted(glob.glob("/dev/serial/by-id/usb-STMicroelectronics_*_if00")) or ["/dev/ttyACM0"]
_PORT = _DEFAULT_PORTS[0]

def set_port(port: str):
    """Optionally set a fixed serial port path."""
    global _PORT
    _PORT = port

def _send_line(payload: str, port: str | None = None, expect_reply=False, timeout=0.8) -> bool:
    p = port or _PORT
    try:
        ser = serial.Serial(p, baudrate=115200, timeout=timeout)
    except Exception as e:
        print(f"[hiko_screen] open error on {p}: {e}")
        return False
    try:
        ser.reset_input_buffer()
        ser.write((payload + "\n").encode("ascii", errors="ignore"))
        ser.flush()
        if expect_reply:
            resp = ser.read(128)
            if resp:
                try:
                    print(resp.decode("ascii", errors="ignore").strip())
                except Exception:
                    pass
        time.sleep(0.03)
        return True
    except Exception as e:
        print(f"[hiko_screen] write error: {e}")
        return False
    finally:
        try: ser.close()
        except Exception: pass

# Convenience wrappers you can import elsewhere
def face(name_or_index: str | int, port: str | None = None) -> bool:
    return _send_line(f"FACE {name_or_index}", port)

def bri(value: int, port: str | None = None) -> bool:
    value = max(0, min(255, int(value)))
    return _send_line(f"BRI {value}", port)

def clr(port: str | None = None) -> bool:
    return _send_line("CLR", port)

# ---- CLI mode (keeps your current behavior) ----
def _main():
    if len(sys.argv) < 2:
        print("Usage: hiko_screen.py [face <name|0..6> | bri <0..255> | clr] [port]")
        sys.exit(1)

    port = None
    if len(sys.argv) >= 4:
        port = sys.argv[3]

    cmd = sys.argv[1].lower()
    if cmd == "face" and len(sys.argv) >= 3:
        ok = face(sys.argv[2], port)
    elif cmd == "bri" and len(sys.argv) >= 3:
        ok = bri(int(sys.argv[2]), port)
    elif cmd == "clr":
        ok = clr(port)
    else:
        print("Invalid command.")
        sys.exit(1)
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    _main()
