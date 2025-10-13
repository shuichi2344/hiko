# touch_bridge.py
import os, time, sys, serial, socket

# Allow override via env; fall back to the by-id you use
PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_6D7433A15248-if00",
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

# If True, also send motor pose commands back to STM32 on tap
MIRROR_POSE_TO_STM32 = os.environ.get("HIKO_MIRROR_POSE", "1") == "1"

def send_cmd(cmd: str):
    """Send a one-line command to the control server (UNIX socket)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            # optional: read short "OK\n"
            try:
                s.settimeout(0.5)
                s.recv(16)
            except Exception:
                pass
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)

def main():
    last_state = None  # None / "REC_START" / "REC_STOP"
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=1) as ser:
                print(f"[touch] listening on {PORT} @ {BAUD}")
                # clear any stale bytes
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "ignore").strip()

                    if line not in ("REC_START", "REC_STOP"):
                        # ignore unrelated MCU logs
                        print(f"[touch] (ign) {line}")
                        continue

                    # drop duplicates (idempotent)
                    if line == last_state:
                        continue
                    last_state = line

                    # forward to control server
                    send_cmd(line)

                    # optional: mirror motor poses to MCU
                    if MIRROR_POSE_TO_STM32:
                        try:
                            if line == "REC_START":
                                ser.write(b"POSE RECORD\n")
                            else:
                                ser.write(b"POSE IDLE\n")
                            ser.flush()
                        except Exception as e:
                            print(f"[touch] serial write error: {e}", file=sys.stderr)

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            # on reconnect, forget last_state (let first edge reflow)
            last_state = None

if __name__ == "__main__":
    main()
