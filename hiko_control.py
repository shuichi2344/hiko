#!/usr/bin/env python3
import os, sys, socket, threading, time, traceback, signal
from typing import Optional, Callable

# ---- Screen serial helpers (your file) ----
from hiko_screen import face as set_face, bri as set_bri, clr as screen_clear, set_port as screen_set_port
# from hiko_screen_shim import face as set_face, bri as set_bri, clr as screen_clear, set_port as screen_set_port

# ===== Config (envs) =====
SOCK_PATH          = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_PORT        = os.environ.get("HIKO_SERIAL_PORT")  # optional override

# One steady idle face (shown whenever nothing else is happening)
IDLE_FACE          = os.environ.get("HIKO_IDLE_FACE", "happy")

# Faces to use for various “activity” phases (override with envs if you like)
REC_FACE           = os.environ.get("HIKO_REC_FACE", "shy")       # while mic is open / recording
TRANSCRIBE_FACE    = os.environ.get("HIKO_TRANS_FACE", "confused")     # while STT/ASR is running
THINK_FACE         = os.environ.get("HIKO_THINK_FACE", "confused")        # while LLM is thinking
TTS_FACE           = os.environ.get("HIKO_TTS_FACE", "speaking")          # while TTS is speaking
ERROR_FACE         = os.environ.get("HIKO_ERROR_FACE", "tired")

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
    # add your own mappings if MCU names differ:
    # "listen": "speaking", "thinking": "tired", etc.
}

def _normalize_face(val: str) -> str:
    v = str(val or "").strip()
    return FACE_ALIASES.get(v.lower(), v)

if SERIAL_PORT:
    screen_set_port(SERIAL_PORT)

class ControlServer(threading.Thread):
    """
    Simple line-based control server over a UNIX socket.
    No idle loop; we keep one steady idle face. Activity commands temporarily
    change the face, then callers should restore the idle face using *_STOP commands.
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

        # current “mode” so you can query if needed later
        self.mode = "IDLE"
        self.idle_face = _normalize_face(IDLE_FACE)

    # ---- lifecycle ----
    def start_server(self):
        # clean stale socket
        try:
            if os.path.exists(self.sock_path):
                os.remove(self.sock_path)
        except Exception:
            pass
        
        if SERIAL_PORT:
            screen_set_port(SERIAL_PORT)

        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        os.chmod(self.sock_path, 0o666)  # world-writable for convenience
        self._srv.listen(8)
        print(f"[control] listening on {self.sock_path}")

        # show idle face immediately
        ok = set_face(self.idle_face)
        print(f"[control] idle face set -> {ok} ({self.idle_face})")
        self.start()  # thread's run()

    def stop_server(self):
        self._stop.set()
        try:
            if self._srv:
                self._srv.close()
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

    # ---- helpers ----
    def _set_mode_face(self, mode: str, face_name: str) -> bool:
        face_norm = _normalize_face(face_name)
        ok = set_face(face_norm)
        if ok:
            self.mode = mode
        return ok

    def _restore_idle(self) -> bool:
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

            # ----- Idle face control -----
            elif cmd == "IDLEFACE":
                # IDLEFACE <name>
                if not args:
                    return f"OK {self.idle_face}"
                newf = _normalize_face(" ".join(args))
                self.idle_face = newf
                # if we are currently idle, reflect immediately
                if self.mode == "IDLE":
                    set_face(self.idle_face)
                return "OK"

            # ----- Recording (mic open) -----
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

            # ----- Transcribing (ASR) -----
            elif cmd == "TRANSCRIBE_START":
                ok = self._set_mode_face("TRANSCRIBE", TRANSCRIBE_FACE)
                print("[control] TRANSCRIBE_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "TRANSCRIBE_STOP":
                ok = self._restore_idle()
                print("[control] TRANSCRIBE_STOP")
                return "OK" if ok else "ERR face failed"

            # ----- Thinking (LLM) -----
            elif cmd == "THINK_START":
                ok = self._set_mode_face("THINK", THINK_FACE)
                print("[control] THINK_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "THINK_STOP":
                ok = self._restore_idle()
                print("[control] THINK_STOP")
                return "OK" if ok else "ERR face failed"

            # ----- Speaking (TTS) -----
            elif cmd == "TTS_START":
                ok = self._set_mode_face("TTS", TTS_FACE)
                print("[control] TTS_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "TTS_STOP":
                ok = self._restore_idle()
                print("[control] TTS_STOP")
                return "OK" if ok else "ERR face failed"

            # ------- Error -------
            elif cmd == "ERROR_START":
                ok = self._set_mode_face("ERROR", ERROR_FACE)
                print("[control] ERROR_START")
                return "OK" if ok else "ERR face failed"

            elif cmd == "ERROR_STOP":
                ok = self._restore_idle()
                print("[control] ERROR_STOP")
                return "OK" if ok else "ERR face failed"

            # ----- Show Idle -------
            elif cmd == "SHOW_IDLE":
                ok = self._restore_idle()
                print("[control] SHOW_IDLE")
                return "OK" if ok else "ERR face failed"

            # ----- Direct controls -----
            elif cmd == "FACE":
                if not args:
                    return "ERR FACE needs <name>"
                val = _normalize_face(" ".join(args))
                ok = set_face(val)
                # do not switch mode; consider it a one-off manual override
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
                # read-only status
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
