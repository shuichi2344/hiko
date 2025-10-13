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

def send_cmd(cmd: str):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)  # don't block forever if server is down
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            # best-effort read of "OK\n"; ignore if not sent
            try:
                s.recv(16)
            except Exception:
                pass
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)

def main():
    last = None  # track last event we forwarded ("REC_START" / "REC_STOP")
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
                            send_cmd("REC_START")
                            last = "REC_START"
                            print("[touch] -> REC_START")
                    elif line in ("REC_STOP", "REC STOP"):
                        if last != "REC_STOP":
                            send_cmd("REC_STOP")
                            last = "REC_STOP"
                            print("[touch] -> REC_STOP")
                    else:
                        # ignore any other MCU logs
                        # print(f"[touch] (ign) {line}")
                        pass

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last = None  # forget state across reconnects

if __name__ == "__main__":
    main()
