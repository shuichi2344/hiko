#!/usr/bin/env python3
# touch_bridge.py
import os, time, sys, serial, socket

# Allow overrides via env; default to your by-id path and socket.
PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_6D7433A15248-if00"
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

def send_cmd(cmd: str) -> bool:
    """Send one command to the control server.
       Returns True only if we get an 'OK' (or 'OK ALREADY')."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.6)  # short, so we don't block the read loop
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            try:
                resp = s.recv(64).decode("utf-8", "ignore").strip()
            except Exception:
                # If server didn't reply, treat as failure so we don't lose edges
                return False
            if resp.startswith("OK"):
                return True
            # Not OK → don't lock in the state; let the next line try again
            print(f"[touch] control replied: {resp}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)
        return False

def main():
    last = None  # track last state we SUCCESSFULLY forwarded ("REC_START"/"REC_STOP")
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
                print(f"[touch] listening on {PORT} @ {BAUD}")
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "ignore").strip()

                    if line in ("REC_START", "REC START"):
                        if last != "REC_START":
                            if send_cmd("REC_START"):
                                last = "REC_START"
                                print("[touch] -> REC_START")
                            else:
                                # Do not change 'last'; we want the next identical line to try again
                                pass

                    elif line in ("REC_STOP", "REC STOP"):
                        if last != "REC_STOP":
                            if send_cmd("REC_STOP"):
                                last = "REC_STOP"
                                print("[touch] -> REC_STOP")
                            else:
                                pass

                    else:
                        # ignore other MCU logs
                        # print(f"[touch] (ign) {line}")
                        pass

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last = None  # reset state across reconnects

if __name__ == "__main__":
    main()
