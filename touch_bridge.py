#!/usr/bin/env python3
# touch_bridge.py — line-based, read-only, auto-recovering
import os, sys, time, socket, serial

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

TOKENS = ("REC_START", "REC_STOP")

def send_cmd(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            resp = s.recv(64).decode("utf-8", "ignore").strip()
            ok = resp.startswith("OK")
            print(f"[touch] -> {cmd} ({'OK' if ok else resp})", flush=True)
            return ok
    except Exception as e:
        print(f"[touch] control send error: {e}", flush=True)
        return False

def main():
    while True:
        try:
            print(f"[touch] opening {PORT} @ {BAUD}", flush=True)
            ser = serial.Serial(PORT, BAUD, timeout=1.0)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            print(f"[touch] listening on {PORT}", flush=True)

            while True:
                try:
                    line = ser.readline().decode("utf-8", "ignore").strip()
                except Exception as e:
                    print(f"[touch] read error: {e}", flush=True)
                    raise  # break to outer reopen loop

                if not line:
                    continue

                # show raw line to debug
                print(f"[touch] raw: {line!r}", flush=True)

                if line in TOKENS:
                    # no dedup — send every event
                    send_cmd(line)

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retry in 1s", flush=True)
            time.sleep(1)
        except OSError as e:
            print(f"[touch] os error: {e}; retry in 1s", flush=True)
            time.sleep(1)
        finally:
            try:
                ser.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
