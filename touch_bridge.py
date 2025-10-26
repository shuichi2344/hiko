#!/usr/bin/env python3
import os, sys, time, socket, serial, re

PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00"
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
TOKEN_RE = re.compile(r"(REC[_ ]START|REC[_ ]STOP)")

def send_cmd(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.8)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            resp = s.recv(128).decode("utf-8", "ignore").strip()
            ok = resp.startswith("OK")
            if not ok:
                print(f"[touch] server resp: {resp}", file=sys.stderr)
            return ok
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)
        return False

def _open():
    ser = serial.Serial(
        PORT, BAUD,
        timeout=0.02,
        inter_byte_timeout=0.02,
    )
    # Handshake so STM32 CDC starts/continues streaming
    try:
        ser.dtr = False; ser.rts = False
        time.sleep(0.05)
        ser.reset_input_buffer(); ser.reset_output_buffer()
        ser.dtr = True; ser.rts = True
        time.sleep(0.02)
    except Exception:
        pass
    print(f"[touch] listening on {PORT} @ {BAUD}")
    return ser

def main():
    last_ok = None
    buf = bytearray()
    while True:
        try:
            with _open() as ser:
                try: ser.reset_input_buffer()
                except Exception: pass
                while True:
                    chunk = ser.read(ser.in_waiting or 1)
                    if not chunk:
                        continue
                    buf.extend(chunk)

                    s = buf.decode("utf-8", "ignore")
                    i = 0
                    while True:
                        m = TOKEN_RE.search(s, i)
                        if not m:
                            break
                        token = m.group(1)
                        i = m.end(1)
                        evt = "REC_START" if "START" in token else "REC_STOP"
                        if evt != last_ok:
                            if send_cmd(evt):
                                last_ok = evt
                                print(f"[touch] -> {evt}")
                            else:
                                print(f"[touch] send failed: {evt}", file=sys.stderr)

                    if i:
                        remain = s[i:]
                        if len(remain) > 64:
                            remain = remain[-64:]
                        buf = bytearray(remain.encode("utf-8", "ignore"))

                    if len(buf) > 2048:
                        buf = buf[-512:]

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last_ok = None
        except Exception as e:
            print(f"[touch] unexpected error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)

if __name__ == "__main__":
    main()
