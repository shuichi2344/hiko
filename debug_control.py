#!/usr/bin/env python3
import os, socket, threading, time, traceback

SOCK_PATH = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

class Control(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.stop = threading.Event()
        self.srv = None

    def start_server(self):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        os.chmod(self.path, 0o666)
        self.srv.listen(8)
        print(f"[control] listening on {self.path}")
        self.start()

    def run(self):
        while not self.stop.is_set():
            try:
                self.srv.settimeout(1.0)
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except Exception:
                print("[control] accept error:\n" + traceback.format_exc())
                continue
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn: socket.socket):
        with conn:
            buf = b""
            while True:
                data = conn.recv(1024)
                if not data: break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8","ignore").strip()
                    print(f"[control] recv: {s!r}")
                    if s in ("REC_START", "REC STOP", "REC_STOP", "PING"):
                        resp = "OK"
                    else:
                        resp = "OK"
                    try: conn.sendall((resp+"\n").encode())
                    except Exception: return

if __name__ == "__main__":
    c = Control(SOCK_PATH)
    c.start_server()
    while True:
        time.sleep(3600)
