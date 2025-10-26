#!/usr/bin/env python3
import os, sys, time, socket, serial, re, threading, traceback, binascii

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_393066403030-if00")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK",  "/tmp/hiko_serial.sock")
TOKEN_RE = re.compile(r"(REC[_ ]START|REC[_ ]STOP)")

def send_to_control(cmd: str, timeout=0.6) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            return s.recv(64).decode("utf-8","ignore").strip().startswith("OK")
    except Exception:
        return False

def _tickle(ser, pulses=1, newline=True):
    try:
        for _ in range(pulses):
            ser.setRTS(True);  time.sleep(0.02);  ser.setRTS(False)
            time.sleep(0.02)
        if newline:
            ser.write(b"\r\n")
    except Exception:
        pass

class SerialBroker:
    def __init__(self, port, baud, serial_sock):
        self.port=port; self.baud=baud; self.serial_sock=serial_sock
        self._stop=threading.Event(); self._ser=None; self._ser_lock=threading.Lock()
        self._last_evt=None

    def start(self):
        try:
            if os.path.exists(self.serial_sock): os.remove(self.serial_sock)
        except Exception: pass

        # ⚠️ Add inter_byte_timeout (helps with partial tokens), and write a wakeup on open
        self._ser = serial.Serial(
            self.port, self.baud,
            timeout=0.02, inter_byte_timeout=0.02,
            rtscts=False, dsrdtr=False, xonxoff=False,
            exclusive=True
        )
        self._ser.setDTR(True)   # required on your board
        self._ser.setRTS(False)
        self._ser.reset_input_buffer(); self._ser.reset_output_buffer()
        # Initial wake-up (mirror of your working script)
        _tickle(self._ser, pulses=2, newline=True)
        print(f"[touch] broker up on {self.port} @ {self.baud} (exclusive, DTR=True, RTS=False)")
        print("[touch] INFO: raw sniffer ENABLED; will dump hex if no tokens are seen.")

        threading.Thread(target=self._read_loop,  daemon=True).start()
        threading.Thread(target=self._serve_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        try:
            if self._ser: self._ser.close()
        except Exception: pass
        try:
            os.remove(self.serial_sock)
        except Exception: pass

    def _read_loop(self):
        buf = bytearray()
        last_tickle = time.time()
        idle_since   = time.time()
        while not self._stop.is_set():
            try:
                now = time.time()
                # periodic 1 s tickle like your mini-test
                if now - last_tickle >= 1.0:
                    last_tickle = now
                    with self._ser_lock:
                        _tickle(self._ser, pulses=1, newline=True)

                # read whatever is available (or 1 byte)
                chunk = self._ser.read(self._ser.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                    idle_since = now
                else:
                    # if we've been idle > 2 s, dump a small hex snippet so we know it's truly quiet
                    if now - idle_since > 2.0 and buf:
                        pass  # already have data buffered; don’t spam
                    elif now - idle_since > 2.0:
                        print("[touch] (quiet >2s) no bytes from MCU")
                    continue

                s = buf.decode("utf-8", "ignore")
                i = 0; matched = False
                while True:
                    m = TOKEN_RE.search(s, i)
                    if not m: break
                    matched = True
                    token = m.group(1); i = m.end(1)
                    evt = "REC_START" if "START" in token else "REC_STOP"

                    if evt != self._last_evt:
                        if send_to_control(evt):
                            self._last_evt = evt
                            print(f"[touch] -> {evt}")
                            # Aggressive keep-alive right after we saw a token
                            with self._ser_lock:
                                _tickle(self._ser, pulses=2, newline=True)
                        else:
                            print(f"[touch] control send failed: {evt}", file=sys.stderr)

                if i:
                    remain = s[i:]
                    if len(remain) > 256: remain = remain[-256:]
                    buf = bytearray(remain.encode("utf-8","ignore"))
                else:
                    # If we got a newline but no token, show the line (helps catch MCU replies)
                    if "\n" in s:
                        line, rest = s.split("\n", 1)
                        line = line.strip()
                        if line:
                            print(f"[touch] non-token line: {line!r}")
                        buf = bytearray(rest.encode("utf-8","ignore"))
                    else:
                        # If there’s binary/partial junk, occasionally dump hex (first 48 bytes)
                        if len(buf) > 0 and len(buf) < 64 and (time.time() % 3.0) < 0.02:
                            print("[touch] partial buffer hex:", binascii.hexlify(buf).decode())

            except serial.SerialException as e:
                print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
                time.sleep(1.0)
            except Exception:
                print("[touch] reader unexpected error:\n" + traceback.format_exc(), file=sys.stderr)
                time.sleep(0.5)

    # --- serve_loop unchanged from the last version you pasted (FACE/BRI/CLR over /tmp/hiko_serial.sock) ---
    def _serve_loop(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.serial_sock); os.chmod(self.serial_sock, 0o666); srv.listen(8)
        print(f"[touch] screen command socket listening on {self.serial_sock}")
        try:
            while not self._stop.is_set():
                try:
                    srv.settimeout(1.0); conn,_ = srv.accept()
                except socket.timeout: continue
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        finally:
            try: srv.close()
            except Exception: pass

    def _handle_client(self, conn: socket.socket):
        with conn:
            data=b""
            try:
                conn.settimeout(2.0)
                while True:
                    chunk=conn.recv(1024)
                    if not chunk: break
                    data+=chunk
                    if b"\n" in data: break
            except Exception: pass
            line=data.decode("utf-8","ignore").strip()
            cmd=(line.split()[:1] or [""])[0].upper()
            if cmd not in ("FACE","BRI","CLR"):
                try: conn.sendall(b"ERR unknown\n")
                except Exception: pass
                return
            try:
                with self._ser_lock:
                    self._ser.write((line + "\n").encode("ascii","ignore"))
                    self._ser.flush()
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
        while True: time.sleep(3600)
    except KeyboardInterrupt: pass
    finally: broker.stop()

if __name__ == "__main__":
    main()
