#!/usr/bin/env python3
import os, sys, time, socket, serial, re

PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00"
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
TOKEN_RE = re.compile(r"(REC[_ ]START|REC[_ ]STOP)")

# If your MCU gates TX on DTR/RTS, this matters:
SER_KW = dict(
    timeout=0.02,
    inter_byte_timeout=0.02,
    rtscts=False,   # disable hw flow control
    dsrdtr=False,   # disable DSR/DTR flow control
    xonxoff=False,  # disable sw flow control
    write_timeout=0
)

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
    ser = serial.Serial(PORT, BAUD, **SER_KW)
    try:
        # Put the CDC into “host is open, ready” state:
        ser.setDTR(True)   # assert DTR (tell device “terminal is open”)
        ser.setRTS(False)  # de-assert RTS (don’t throttle device TX)
        time.sleep(0.02)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    print(f"[touch] listening on {PORT} @ {BAUD}")
    return ser

def main():
    last_ok = None
    buf = bytearray()
    last_any = time.monotonic()   # last time we saw any bytes

    while True:
        try:
            with _open() as ser:
                while True:
                    chunk = ser.read(ser.in_waiting or 1)
                    if chunk:
                        last_any = time.monotonic()
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
                            # keep tail small
                            if len(remain) > 64:
                                remain = remain[-64:]
                            buf = bytearray(remain.encode("utf-8", "ignore"))

                        if len(buf) > 2048:
                            buf = buf[-512:]

                    # If the device gets “quiet” due to control-line funkiness, re-open to re-assert DTR/RTS
                    if (time.monotonic() - last_any) > 3.0:
                        print("[touch] idle >3s; re-open to reassert DTR/RTS")
                        break  # break inner loop → __exit__ closes → we reopen

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last_ok = None
        except Exception as e:
            print(f"[touch] unexpected error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)

if __name__ == "__main__":
    main()
