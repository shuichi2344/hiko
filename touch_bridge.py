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
            s.settimeout(1.0)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd + "\n").encode())
            return s.recv(64).decode(errors="ignore").strip().startswith("OK")
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)
        return False

def main():
    last = None
    with serial.Serial(
        PORT, BAUD, timeout=0.02,
        rtscts=False, dsrdtr=False, xonxoff=False,
        exclusive=True  # avoid multi-open races
    ) as ser:
        # Required combo for your board:
        ser.setDTR(True)
        ser.setRTS(False)
        ser.reset_input_buffer(); ser.reset_output_buffer()
        print(f"[touch] listening (DTR=True, RTS=False) on {PORT} @ {BAUD}")
        buf = bytearray()
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
                if evt != last:
                    if send_cmd(evt):
                        last = evt
                        print(f"[touch] -> {evt}")
                    else:
                        print(f"[touch] send failed: {evt}", file=sys.stderr)
            if i:
                # Keep buffer bounded
                remain = s[i:]
                if len(remain) > 128:
                    remain = remain[-128:]
                buf = bytearray(remain.encode("utf-8", "ignore"))

if __name__ == "__main__":
    main()
