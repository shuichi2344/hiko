#!/usr/bin/env python3
# touch_bridge.py — robust token scanner (newline-independent)
import os, time, sys, serial, socket

# Env overrides if needed
PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00"
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

# Accept both styles just in case
TOKENS = ("REC_START", "REC START", "REC_STOP", "REC STOP")

def send_cmd(cmd: str) -> bool:
    """Send one command to the control server; True only on OK/OK ALREADY."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.6)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            try:
                resp = s.recv(64).decode("utf-8", "ignore").strip()
            except Exception:
                return False
            return resp.startswith("OK")
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)
        return False

def _extract_events(buf: bytearray):
    """Yield REC_* events found in buffer; trim consumed bytes."""
    out = []
    s = buf.decode("utf-8", "ignore")
    i = 0
    while True:
        # find next occurrence of any token from position i
        found_positions = [s.find(t, i) for t in TOKENS]
        found_positions = [p for p in found_positions if p != -1]
        if not found_positions:
            break
        pos = min(found_positions)
        # pick the longest token that matches at pos (handles START vs STOP overlap)
        matched = max([t for t in TOKENS if s.startswith(t, pos)], key=len)
        out.append("REC_START" if "START" in matched else "REC_STOP")
        i = pos + len(matched)

    if i:
        del buf[:i]  # drop bytes we've scanned past
    # keep buffer bounded even if junk streams in
    if len(buf) > 1024:
        del buf[:-256]
    return out

def main():
    last_ok = None  # last state ACKed by control server: "REC_START" or "REC_STOP"
    while True:
        try:
            with serial.Serial(
                PORT,
                BAUD,
                timeout=0.02,             # fast loop
                inter_byte_timeout=0.02,   # break out if stream stalls mid-line
            ) as ser:
                print(f"[touch] listening on {PORT} @ {BAUD}")
                try: ser.reset_input_buffer()
                except Exception: pass

                buf = bytearray()
                while True:
                    # slurp whatever is available; if nothing, read 1 byte (non-blocking-ish)
                    try:
                        chunk = ser.read(ser.in_waiting or 1)
                    except Exception as e:
                        print(f"[touch] read error: {e}", file=sys.stderr)
                        break
                    if not chunk:
                        continue

                    buf.extend(chunk)

                    for evt in _extract_events(buf):
                        # drop duplicates only if we previously succeeded
                        if evt == last_ok:
                            continue
                        if send_cmd(evt):
                            last_ok = evt
                            print(f"[touch] -> {evt}")
                        else:
                            # don’t update last_ok; retry on next same token
                            pass

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last_ok = None  # forget state across reconnects

if __name__ == "__main__":
    main()
