#!/usr/bin/env python3
import os, sys, time, socket, serial, re, threading, traceback

# ===== Config =====
PORT = os.environ.get(
    "HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00"
)
BAUD = 115200

# Control server (hiko_control.py) — we SEND REC_START/STOP to this
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

# New: we also run our OWN socket to accept screen commands (FACE/BRI/CLR)
SERIAL_SOCK = os.environ.get("HIKO_SERIAL_SOCK", "/tmp/hiko_serial.sock")

TOKEN_RE = re.compile(r"(REC[_ ]START|REC[_ ]STOP)")

def send_to_control(cmd: str, timeout=0.6) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            resp = s.recv(64).decode("utf-8", "ignore").strip()
            return resp.startswith("OK")
    except Exception:
        return False

def rts_tickle(ser):
    try:
        ser.setRTS(True);  time.sleep(0.02);  ser.setRTS(False)
        ser.write(b"\n")  # harmless keepalive/wakeup
    except Exception:
        pass

class SerialBroker:
    """
    Owns the STM32 CDC port exclusively.
    - Thread A: reads tokens and notifies hiko_control via CONTROL_SOCK.
    - Thread B: serves SERIAL_SOCK to receive FACE/BRI/CLR and writes them to serial.
    """
    def __init__(self, port: str, baud: int, serial_sock: str):
        self.port = port
        self.baud = baud
        self.serial_sock = serial_sock
        self._stop = threading.Event()
        self._ser = None
        self._ser_lock = threading.Lock()
        self._last_evt = None

    # ---------- lifecycle ----------
    def start(self):
        # clean stale socket path
        try:
            if os.path.exists(self.serial_sock):
                os.remove(self.serial_sock)
        except Exception:
            pass

        # open once, exclusive
        self._ser = serial.Serial(
            self.port, self.baud,
            timeout=0.02,
            rtscts=False, dsrdtr=False, xonxoff=False,
            exclusive=True
        )
        # Required on your board
        self._ser.setDTR(True)
        self._ser.setRTS(False)
        self._ser.reset_input_buffer(); self._ser.reset_output_buffer()
        print(f"[touch] broker up on {self.port} @ {self.baud} (exclusive, DTR=True, RTS=False)")

        # start threads
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._server = threading.Thread(target=self._serve_loop, daemon=True)
        self._server.start()

    def stop(self):
        self._stop.set()
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        try:
            os.remove(self.serial_sock)
        except Exception:
            pass

    # ---------- token reader ----------
    def _read_loop(self):
        buf = bytearray()
        last_tickle = time.time()
        while not self._stop.is_set():
            try:
                # periodic keepalive/tickle ~1s
                now = time.time()
                if now - last_tickle >= 1.0:
                    last_tickle = now
                    with self._ser_lock:
                        rts_tickle(self._ser)

                # read bytes
                chunk = self._ser.read(self._ser.in_waiting or 1)
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
                    if evt != self._last_evt:
                        ok = send_to_control(evt)
                        if ok:
                            self._last_evt = evt
                            print(f"[touch] -> {evt}")
                            with self._ser_lock:
                                rts_tickle(self._ser)  # edge after notifying host
                        else:
                            print(f"[touch] control send failed: {evt}", file=sys.stderr)

                if i:
                    remain = s[i:]
                    if len(remain) > 256:
                        remain = remain[-256:]
                    buf = bytearray(remain.encode("utf-8", "ignore"))

            except serial.SerialException as e:
                print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
                time.sleep(1.0)
            except Exception:
                print("[touch] reader unexpected error:\n" + traceback.format_exc(), file=sys.stderr)
                time.sleep(0.5)

    # ---------- screen command server ----------
    def _serve_loop(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.serial_sock)
        os.chmod(self.serial_sock, 0o666)
        srv.listen(8)
        print(f"[touch] screen command socket listening on {self.serial_sock}")
        try:
            while not self._stop.is_set():
                try:
                    srv.settimeout(1.0)
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except Exception:
                    if not self._stop.is_set():
                        print("[touch] accept error:\n" + traceback.format_exc(), file=sys.stderr)
                    continue
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        finally:
            try: srv.close()
            except Exception: pass

    def _handle_client(self, conn: socket.socket):
        with conn:
            data = b""
            try:
                conn.settimeout(2.0)
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break
            except Exception:
                pass

            line = data.decode("utf-8", "ignore").strip()
            # Accept only these commands to forward to MCU
            # FACE <name|idx> | BRI <0..255> | CLR
            if not line:
                try: conn.sendall(b"ERR empty\n")
                except Exception: pass
                return

            cmd = line.split()[0].upper()
            if cmd not in ("FACE", "BRI", "CLR"):
                try: conn.sendall(b"ERR unknown\n")
                except Exception: pass
                return

            try:
                with self._ser_lock:
                    # write line to serial and add newline
                    self._ser.write((line + "\n").encode("ascii", "ignore"))
                    self._ser.flush()
                    # small settle
                    time.sleep(0.02)
                try: conn.sendall(b"OK\n")
                except Exception: pass
            except Exception as e:
                try: conn.sendall(f"ERR {e}\n".encode("utf-8"))
                except Exception: pass

def main():
    broker = SerialBroker(PORT, BAUD, SERIAL_SOCK)
    broker.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        broker.stop()

if __name__ == "__main__":
    main()
