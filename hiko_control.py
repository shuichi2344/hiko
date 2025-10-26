#!/usr/bin/env python3
import os, sys, socket, threading, time, traceback, signal
from typing import Optional, Callable

# ===== Config (envs) =====
SOCK_PATH   = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK = os.environ.get("HIKO_SERIAL_SOCK", "/tmp/hiko_serial.sock")

# Faces
IDLE_FACE       = os.environ.get("HIKO_IDLE_FACE", "happy")
REC_FACE        = os.environ.get("HIKO_REC_FACE", "shy")
TRANSCRIBE_FACE = os.environ.get("HIKO_TRANS_FACE", "confused")
THINK_FACE      = os.environ.get("HIKO_THINK_FACE", "confused")
TTS_FACE        = os.environ.get("HIKO_TTS_FACE", "speaking")
ERROR_FACE      = os.environ.get("HIKO_ERROR_FACE", "tired")

FACE_ALIASES = {
    "neutral":"neutral","happy":"happy","flower":"flower","shy":"shy","cat":"cat",
    "confused":"confused","speaking":"speaking","tired":"tired","wink":"wink",
}

def _normalize_face(val: str) -> str:
    v = str(val or "").strip()
    return FACE_ALIASES.get(v.lower(), v)

# === IMPORTANT FLAG ===
DISABLE_FACES = os.getenv("HIKO_DISABLE_FACES", "0") == "1"

# ---- tiny client to touch_bridge's serial socket ----
def _serial_cmd(line: str, timeout=0.8) -> bool:
    if DISABLE_FACES:
        return True
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(SERIAL_SOCK)
            s.sendall((line.strip() + "\n").encode("utf-8"))
            resp = s.recv(128).decode("utf-8", "ignore").strip()
            return resp.startswith("OK")
    except Exception as e:
        # Do not spam when faces are disabled; only log when enabled
        print(f"[control] serial sock error: {e}")
        return False

def set_face(name_or_index: str) -> bool:
    return _serial_cmd(f"FACE {_normalize_face(name_or_index)}")

def set_bri(value: int) -> bool:
    try:
        v = max(0, min(255, int(value)))
    except Exception:
        return False
    return _serial_cmd(f"BRI {v}")

def screen_clear() -> bool:
    return _serial_cmd("CLR")

class ControlServer(threading.Thread):
    """
    Simple line-based control server over a UNIX socket.
    Keeps an idle face; phase commands swap face temporarily.
    """
    def __init__(
        self,
        sock_path: str,
        on_rec_start: Optional[Callable[[], None]] = None,
        on_rec_stop: Optional[Callable[[], None]] = None,
    ):
        super().__init__(daemon=True)
        self.sock_path = sock_path
        self._srv = None
        self._stop = threading.Event()
        self.on_rec_start = on_rec_start
        self.on_rec_stop  = on_rec_stop
        self.mode = "IDLE"
        self.idle_face = _normalize_face(IDLE_FACE)

    # ---- lifecycle ----
    def start_server(self):
        try:
            if os.path.exists(self.sock_path):
                os.remove(self.sock_path)
        except Exception:
            pass

        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        os.chmod(self.sock_path, 0o666)
        self._srv.listen(8)
        print(f"[control] listening on {self.sock_path}")

        # Only try to set face if faces are enabled
        if not DISABLE_FACES:
            ok = set_face(self.idle_face)
            print(f"[control] idle face set -> {ok} ({self.idle_face})")
        else:
            print(f"[control] faces disabled (HIKO_DISABLE_FACES=1)")

        self.start()

    def stop_server(self):
        self._stop.set()
        try:
            if self._srv: self._srv.close()
        except Exception:
            pass
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
                try:
                    data = conn.recv(1024)
                except Exception:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    print(f"[control] recv:", line)
                    resp = self._dispatch(line.decode("utf-8", "ignore").strip())
                    try:
                        conn.sendall((resp + "\n").encode("utf-8"))
                        return
                    except Exception:
                        return

    def _set_mode_face(self, mode: str, face_name: str) -> bool:
        # If faces disabled, just switch mode and pretend OK
        if DISABLE_FACES:
            self.mode = mode
            return True
        face_norm = _normalize_face(face_name)
        ok = set_face(face_norm)
        if ok:
            self.mode = mode
        return ok

    def _restore_idle(self) -> bool:
        if DISABLE_FACES:
            self.mode = "IDLE"
            return True
        ok = set_face(self.idle_face)
        if ok:
            self.mode = "IDLE"
        return ok

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

            elif cmd == "IDLEFACE":
                if not args:
                    return f"OK {self.idle_face}"
                newf = _normalize_face(" ".join(args))
                self.idle_face = newf
                if self.mode == "IDLE" and not DISABLE_FACES:
                    set_face(self.idle_face)
                return "OK"

            elif cmd == "REC_START":
                if self.on_rec_start: self.on_rec_start()
                ok = self._set_mode_face("REC", REC_FACE)
                print("[control] REC_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "REC_STOP":
                if self.on_rec_stop: self.on_rec_stop()
                ok = self._restore_idle()
                print("[control] REC_STOP")
                return "OK" if ok else "ERR face failed"

            elif cmd == "TRANSCRIBE_START":
                ok = self._set_mode_face("TRANSCRIBE", TRANSCRIBE_FACE)
                print("[control] TRANSCRIBE_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "TRANSCRIBE_STOP":
                ok = self._restore_idle()
                print("[control] TRANSCRIBE_STOP")
                return "OK" if ok else "ERR face failed"

            elif cmd == "THINK_START":
                ok = self._set_mode_face("THINK", THINK_FACE)
                print("[control] THINK_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "THINK_STOP":
                ok = self._restore_idle()
                print("[control] THINK_STOP")
                return "OK" if ok else "ERR face failed"

            elif cmd == "TTS_START":
                ok = self._set_mode_face("TTS", TTS_FACE)
                print("[control] TTS_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "TTS_STOP":
                ok = self._restore_idle()
                print("[control] TTS_STOP")
                return "OK" if ok else "ERR face failed"

            elif cmd == "ERROR_START":
                ok = self._set_mode_face("ERROR", ERROR_FACE)
                print("[control] ERROR_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "ERROR_STOP":
                ok = self._restore_idle()
                print("[control] ERROR_STOP")
                return "OK" if ok else "ERR face failed"

            elif cmd == "SHOW_IDLE":
                ok = self._restore_idle()
                print("[control] SHOW_IDLE")
                return "OK" if ok else "ERR face failed"

            elif cmd == "FACE":
                if not args:
                    return "ERR FACE needs <name>"
                ok = set_face(" ".join(args))
                return "OK" if ok else "ERR face failed"

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

            elif cmd == "MODE":
                return f"OK {self.mode}"

            else:
                return f"ERR unknown '{cmd}'"
        except Exception as e:
            return f"ERR {e}"

# --------- CLI entry ---------
def _standalone():
    srv = ControlServer(
        sock_path=SOCK_PATH,
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
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    _standalone()
