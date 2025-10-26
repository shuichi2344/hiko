#!/usr/bin/env python3
# touch_bridge.py — single-owner broker:
#  - Owns the STM32 USB serial (read+write, exclusive)
#  - Forwards REC_START/REC_STOP to /tmp/hiko_control.sock
#  - Exposes /tmp/hiko_serial.sock to accept FACE/BRI/CLR and writes to MCU

import os, sys, time, socket, threading
import serial

PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00"
)
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SCREEN_SOCK  = os.environ.get("HIKO_SERIAL_SOCK", "/tmp/hiko_serial.sock")

TOKENS = ("REC_START","REC_STOP")

# ---------- Control socket client (fire-and-forget) ----------
def _send_to_control(cmd: str) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip()+"\n").encode("utf-8"))
            # ignore reply; we do not block the stream on ACKs
            try: s.recv(64)
            except Exception: pass
    except Exception:
        pass  # control server might not be up yet; that's fine

# ---------- Broker server for faces ----------
class ScreenServer(threading.Thread):
    def __init__(self, ser, lock):
        super().__init__(daemon=True)
        self.ser = ser          # shared serial object
        self.lock = lock        # to serialize writes
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        try:
            os.remove(SCREEN_SOCK)
        except Exception:
            pass

    def run(self):
        # clean stale socket
        try:
            if os.path.exists(SCREEN_SOCK):
                os.remove(SCREEN_SOCK)
        except Exception:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(SCREEN_SOCK)
        os.chmod(SCREEN_SOCK, 0o666)
        srv.listen(8)
        print(f"[touch] screen command socket listening on {SCREEN_SOCK}")

        while not self._stop.is_set():
            try:
                srv.settimeout(1.0)
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                if not self._stop.is_set():
                    print("[touch] screen socket accept error", file=sys.stderr)
                continue

            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn):
        with conn:
            try:
                data = conn.recv(256)
            except Exception:
                return
            if not data:
                return
            line = data.decode("utf-8","ignore").strip()
            # Allow: FACE <name>, BRI <0..255>, CLR
            if not line:
                _reply(conn, "ERR empty"); return

            cmd, *rest = line.split()
            cmd = cmd.upper()
            if cmd == "FACE" and rest:
                payload = f"FACE {' '.join(rest)}\n"
            elif cmd == "BRI" and rest:
                try:
                    level = max(0, min(255, int(rest[0])))
                except Exception:
                    _reply(conn, "ERR bad BRI"); return
                payload = f"BRI {level}\n"
            elif cmd == "CLR":
                payload = "CLR\n"
            else:
                _reply(conn, f"ERR unknown '{cmd}'"); return

            # Write to MCU
            try:
                with self.lock:
                    self.ser.write(payload.encode("ascii", errors="ignore"))
                    self.ser.flush()
                _reply(conn, "OK")
            except Exception as e:
                _reply(conn, f"ERR {e}")

def _reply(conn, msg: str):
    try:
        conn.sendall((msg+"\n").encode("utf-8"))
    except Exception:
        pass

# ---------- token scanner ----------
def _extract_events(buf: bytearray):
    out = []
    s = buf.decode("utf-8","ignore")
    i = 0
    while True:
        pos_candidates = [s.find(t, i) for t in TOKENS]
        pos_candidates = [p for p in pos_candidates if p != -1]
        if not pos_candidates:
            break
        pos = min(pos_candidates)
        matched = max([t for t in TOKENS if s.startswith(t, pos)], key=len)
        out.append("REC_START" if matched.endswith("START") else "REC_STOP")
        i = pos + len(matched)
    if i:
        del buf[:i]
    if len(buf) > 1024:
        del buf[:-256]
    return out

def main():
    while True:
        try:
            ser = serial.Serial(
                PORT,
                BAUD,
                timeout=0.02,
                inter_byte_timeout=0.02,
                exclusive=True
            )
            # Set lines for some STM32 VCPs (optional but harmless)
            try:
                ser.dtr = True
                ser.rts = False
            except Exception:
                pass

            print(f"[touch] broker up on {PORT} @ {BAUD} (exclusive, DTR=True, RTS=False)")

            try: ser.reset_input_buffer()
            except Exception: pass

            lock = threading.Lock()
            screen_srv = ScreenServer(ser, lock)
            screen_srv.start()

            buf = bytearray()
            last_evt = None
            last_time = 0.0

            while True:
                try:
                    chunk = ser.read(ser.in_waiting or 1)
                except Exception as e:
                    print(f"[touch] read error: {e}", file=sys.stderr)
                    break
                if not chunk:
                    continue
                buf.extend(chunk)

                now = time.monotonic()
                for evt in _extract_events(buf):
                    # simple debouncing + no-duplicate streak
                    if evt == last_evt and (now - last_time) < 0.10:
                        continue
                    last_evt = evt
                    last_time = now
                    _send_to_control(evt)
                    print(f"[touch] -> {evt}")

        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
        except Exception as e:
            print(f"[touch] fatal: {e}; restarting in 1s", file=sys.stderr)
            time.sleep(1)
        finally:
            try:
                screen_srv.stop()
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
