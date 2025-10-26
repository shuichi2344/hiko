# hiko_screen.py
import sys, time, glob
import serial  # pip install pyserial

_DEFAULT_PORTS = sorted(glob.glob("/dev/serial/by-id/usb-STMicroelectronics_*_if00")) or ["/dev/ttyACM0"]
_PORT = _DEFAULT_PORTS[0]

def set_port(port: str):
    global _PORT
    _PORT = port

def _send_line(payload: str, port: str | None = None, expect_reply=False, timeout=0.8) -> bool:
    p = port or _PORT
    try:
        # IMPORTANT: make this open exclusive so we don't steal the port while touch_bridge owns it
        ser = serial.Serial(p, baudrate=115200, timeout=timeout, exclusive=True)
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
