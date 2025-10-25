#!/usr/bin/env python3
import os, sys, time, socket, serial

PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00"
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

TOKENS = {"REC_START", "REC_STOP"}

def send_cmd(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd + "\n").encode("utf-8"))
            resp = s.recv(128).decode("utf-8", "ignore").strip()
            return resp.startswith("OK")
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)
        return False

def main():
    last_ok = None
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
                print(f"[touch] listening on {PORT} @ {BAUD}")
                try: ser.reset_input_buffer()
                except Exception: pass

                while True:
                    line = ser.readline().decode("utf-8", "ignore").strip()
                    if not line:
                        continue

                    # Fast-path: exact match on our two tokens
                    if line in TOKENS:
                        if line == last_ok:
                            continue  # de-dupe same state
                        ok = send_cmd(line)
                        if ok:
                            last_ok = line
                            print(f"[touch] -> {line}")
                        else:
                            print(f"[touch] send failed: {line}", file=sys.stderr)
                    else:
                        # ignore noise but log for visibility
                        print(f"[touch] ignore: {line}")

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last_ok = None

if __name__ == "__main__":
    main()
