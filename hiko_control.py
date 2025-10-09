#!/usr/bin/env python3
import os, sys, socket, threading, time, traceback, signal
from pathlib import Path
from typing import Callable, Optional, List, Union

# ---- Screen serial helpers (your file) ----
from hiko_screen import face as set_face, bri as set_bri, clr as screen_clear, set_port as screen_set_port

SOCK_PATH = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_PORT = os.environ.get("HIKO_SERIAL_PORT")  # optional override
IDLE_INTERVAL = float(os.environ.get("HIKO_IDLE_INTERVAL", "4.0"))   # seconds between idle face swaps
IDLE_FACES_ENV = os.environ.get("HIKO_IDLE_FACES") 
DEFAULT_IDLE_FACES: List[str] = ["happy", "neutral", "flower", "shy"] 
FACE_ALIASES = {
    "neutral": "neutral",
    "happy":   "happy",
    "flower":  "flower",
    "shy":     "shy",
    "cat": "cat",
    "confused": "confused",
    "speaking": "speaking",
    "tired": "tired",
    "wink": "wink",
    # add more if you ever rename faces on the MCU:
    # "talk": "speaking", "smile": "happy", etc.
}

def _normalize_face(val: str) -> str:
    v = str(val).strip()
    return FACE_ALIASES.get(v.lower(), v) 

if SERIAL_PORT:
    screen_set_port(SERIAL_PORT)

def _parse_idle_faces(env_val: Optional[str]) -> List[str]:
    if not env_val:
        return DEFAULT_IDLE_FACES
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    return parts or DEFAULT_IDLE_FACES

# ---------- Idle Face Looper ----------
class IdleLooper(threading.Thread):
    def __init__(self, faces: List[str], interval: float = 4.0):
        super().__init__(daemon=True)
        self.faces = faces
        self.interval = max(0.5, interval)
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._hold_until = 0.0  # epoch when pin/hold expires

    def enable(self):
        self._enabled.set()

    def disable(self):
        self._enabled.clear()

    def stop(self):
        self._stop.set()

    def hold(self, seconds: float, face_value: str):
        with self._lock:
            val = _normalize_face(face_value)
            set_face(face_value)
            self._hold_until = time.time() + max(0.0, seconds)

    def run(self):
        i = 0
        while not self._stop.is_set():
            if not self._enabled.is_set():
                time.sleep(0.1)
                continue

            # If a HOLD is active, just wait until it expires
            with self._lock:
                now = time.time()
                if now < self._hold_until:
                    time.sleep(0.1)
                    continue

            # advance face
            val = _normalize_face(self.faces[i % len(self.faces)])
            set_face(val)
            i += 1
            for _ in range(int(self.interval / 0.1)):
                if self._stop.is_set(): break
                # respect incoming holds immediately
                with self._lock:
                    if time.time() < self._hold_until:
                        break
                time.sleep(0.1)

# ---------- Control Server ----------
class ControlServer(threading.Thread):
    """
    Simple line-based control server over a UNIX socket.
    Each client line => a command. Responds with 'OK' or 'ERR <msg>'.

    Hook into on_rec_start/on_rec_stop to integrate with your chatbot.
    """
    def __init__(
        self,
        sock_path: str,
        idle_faces: List[str],
        idle_interval: float,
        on_rec_start: Optional[Callable[[], None]] = None,
        on_rec_stop: Optional[Callable[[], None]] = None,
    ):
        super().__init__(daemon=True)
        self.sock_path = sock_path
        self.on_rec_start = on_rec_start
        self.on_rec_stop = on_rec_stop
        self.idle = IdleLooper(idle_faces, idle_interval)
        self._srv = None
        self._stop = threading.Event()

    # ---- lifecycle ----
    def start_server(self):
        # clean stale socket
        try:
            if os.path.exists(self.sock_path):
                os.remove(self.sock_path)
        except Exception:
            pass

        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        os.chmod(self.sock_path, 0o666)  # world-writable for convenience
        self._srv.listen(8)
        print(f"[control] listening on {self.sock_path}")

        # start idle looper enabled by default
        self.idle.enable()
        self.idle.start()

        self.start()  # thread's run()

    def stop_server(self):
        self._stop.set()
        try:
            if self._srv:
                self._srv.close()
        except Exception:
            pass
        self.idle.stop()
        try:
            os.remove(self.sock_path)
        except Exception:
            pass

    # ---- thread main ----
    def run(self):
        while not self._stop.is_set():
            try:
                self._srv.settimeout(1.0)
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except Exception:
                if not self._stop.is_set():
                    print("[control] accept error:\n" + traceback.format_exc())
                continue
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket):
        with conn:
            buf = b""
            while True:
                data = b""
                try:
                    data = conn.recv(1024)
                except Exception:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    resp = self._dispatch(line.decode("utf-8", "ignore").strip())
                    try:
                        conn.sendall((resp + "\n").encode("utf-8"))
                    except Exception:
                        return

    # ---- commands ----
    def _dispatch(self, line: str) -> str:
        if not line:
            return "ERR empty"

        parts = line.split()
        cmd = parts[0].upper()
        args = parts[1:]

        try:
            if cmd == "PING":
                return f"OK PONG {int(time.time())}"

            elif cmd == "REC_START":
                # disable idle while recording (optional, feels nicer)
                self.idle.disable()
                if self.on_rec_start:
                    self.on_rec_start()
                print("[control] REC_START")
                return "OK"

            elif cmd == "REC_STOP":
                if self.on_rec_stop:
                    self.on_rec_stop()
                # re-enable idle after recording
                self.idle.enable()
                print("[control] REC_STOP")
                return "OK"

            elif cmd == "FACE":
                if not args:
                    return "ERR FACE needs <name|index>"
                val = _normalize_face(" ".join(args))
                ok = set_face(val)
                if ok:
                    # brief hold so idle doesn't immediately overwrite
                    self.idle.hold(1.0, val)
                    return "OK"
                return "ERR face failed"

            elif cmd == "BRI":
                if not args:
                    return "ERR BRI needs <0..255>"
                try:
                    level = max(0, min(255, int(args[0])))
                except ValueError:
                    return "ERR bad BRI"
                ok = set_bri(level)
                return "OK" if ok else "ERR bri failed"

            elif cmd == "CLR":
                ok = screen_clear()
                return "OK" if ok else "ERR clr failed"

            elif cmd == "IDLE":
                if not args:
                    return "ERR IDLE needs ON|OFF"
                state = args[0].upper()
                if state == "ON":
                    self.idle.enable()
                    return "OK"
                elif state == "OFF":
                    self.idle.disable()
                    return "OK"
                else:
                    return "ERR IDLE needs ON|OFF"

            elif cmd == "HOLD":
                # HOLD <seconds> <face...>
                if len(args) < 2:
                    return "ERR HOLD <seconds> <face>"
                seconds = float(args[0])
                val = _normalize_face(" ".join(args[1:]))
                self.idle.hold(seconds, val)
                return "OK"

            else:
                return f"ERR unknown '{cmd}'"
        except Exception as e:
            return f"ERR {e}"

# --------- CLI entry ---------
def _standalone():
    faces = _parse_idle_faces(IDLE_FACES_ENV)
    srv = ControlServer(
        sock_path=SOCK_PATH,
        idle_faces=faces,
        idle_interval=IDLE_INTERVAL,
        on_rec_start=None,
        on_rec_stop=None,
    )
    def _shutdown(*_):
        print("\n[control] shutting down...")
        srv.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    srv.start_server()
    # sleep forever
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    _standalone()
