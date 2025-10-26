#!/usr/bin/env python3
import os, socket, threading, time, traceback, signal, sys
from typing import Optional, Callable

SOCK_PATH   = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK = os.environ.get("HIKO_SERIAL_SOCK", "/tmp/hiko_serial.sock")

IDLE_FACE       = os.environ.get("HIKO_IDLE_FACE", "happy")
REC_FACE        = os.environ.get("HIKO_REC_FACE", "shy")
TRANSCRIBE_FACE = os.environ.get("HIKO_TRANS_FACE", "confused")
THINK_FACE      = os.environ.get("HIKO_THINK_FACE", "confused")
TTS_FACE        = os.environ.get("HIKO_TTS_FACE", "speaking")
ERROR_FACE      = os.environ.get("HIKO_ERROR_FACE", "tired")

FACE_ALIASES = {"neutral":"neutral","happy":"happy","flower":"flower","shy":"shy","cat":"cat",
                "confused":"confused","speaking":"speaking","tired":"tired","wink":"wink"}

def _norm(v:str)->str: return FACE_ALIASES.get((v or "").strip().lower(), v)

def _serial_cmd(line: str, timeout=0.6) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(SERIAL_SOCK)
            s.sendall((line.strip()+"\n").encode("utf-8"))
            resp = s.recv(128).decode("utf-8","ignore").strip()
            return resp.startswith("OK")
    except Exception:
        # keep quiet; the caller can decide what to do
        return False

def set_face(name:str)->bool: return _serial_cmd(f"FACE {_norm(name)}")
def set_bri(v:int)->bool:
    try: v=max(0,min(255,int(v)))
    except: return False
    return _serial_cmd(f"BRI {v}")
def screen_clear()->bool: return _serial_cmd("CLR")

class ControlServer(threading.Thread):
    def __init__(self, sock_path: str, on_rec_start: Optional[Callable]=None, on_rec_stop: Optional[Callable]=None):
        super().__init__(daemon=True)
        self.sock_path = sock_path
        self._srv=None
        self._stop=threading.Event()
        self.on_rec_start=on_rec_start
        self.on_rec_stop=on_rec_stop
        self.mode="IDLE"
        self.idle_face=_norm(IDLE_FACE)

    def start_server(self):
        try:
            if os.path.exists(self.sock_path): os.remove(self.sock_path)
        except: pass
        self._srv=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        os.chmod(self.sock_path, 0o666)
        self._srv.listen(8)
        print(f"[control] listening on {self.sock_path}")
        # wait up to ~2s for the broker socket to be ready, then set the idle face
        deadline = time.time() + 2.0
        ok = False
        while time.time() < deadline:
            if os.path.exists(SERIAL_SOCK):
                ok = set_face(self.idle_face)
                if ok:
                    break
            time.sleep(0.1)
        print(f"[control] idle face set -> {ok} ({self.idle_face})")
        self.start()

    def stop_server(self):
        self._stop.set()
        try:
            if self._srv: self._srv.close()
        except: pass
        try: os.remove(self.sock_path)
        except: pass

    def run(self):
        while not self._stop.is_set():
            try:
                self._srv.settimeout(1.0)
                conn,_=self._srv.accept()
            except socket.timeout:
                continue
            except Exception:
                if not self._stop.is_set():
                    print("[control] accept error:\n"+traceback.format_exc())
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        with conn:
            buf=b""
            while True:
                try: data=conn.recv(1024)
                except: break
                if not data: break
                buf+=data
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1)
                    print(f"[control] recv:", line)
                    resp=self._dispatch(line.decode("utf-8","ignore").strip())
                    try: conn.sendall((resp+"\n").encode("utf-8")); return
                    except: return

    def _set_mode_face(self, mode, face)->bool:
        ok=set_face(face)
        if ok: self.mode=mode
        return ok

    def _restore_idle(self)->bool:
        ok=set_face(self.idle_face)
        if ok: self.mode="IDLE"
        return ok

    def _dispatch(self, line:str)->str:
        if not line: return "ERR empty"
        cmd,*args=line.split()
        cmd=cmd.upper()
        try:
            if cmd=="PING": return f"OK PONG {int(time.time())}"
            elif cmd=="IDLEFACE":
                if not args: return f"OK {self.idle_face}"
                self.idle_face=_norm(" ".join(args))
                if self.mode=="IDLE": set_face(self.idle_face)
                return "OK"
            elif cmd=="REC_START":
                if self.on_rec_start: self.on_rec_start()
                ok=self._set_mode_face("REC", _norm(REC_FACE)); print("[control] REC_START")
                return "OK" if ok else "ERR"
            elif cmd=="REC_STOP":
                if self.on_rec_stop: self.on_rec_stop()
                ok=self._restore_idle(); print("[control] REC_STOP")
                return "OK" if ok else "ERR"
            elif cmd=="TRANSCRIBE_START":
                ok=self._set_mode_face("TRANSCRIBE", _norm(TRANSCRIBE_FACE)); print("[control] TRANSCRIBE_START")
                return "OK" if ok else "ERR"
            elif cmd=="TRANSCRIBE_STOP":
                ok=self._restore_idle(); print("[control] TRANSCRIBE_STOP")
                return "OK" if ok else "ERR"
            elif cmd=="THINK_START":
                ok=self._set_mode_face("THINK", _norm(THINK_FACE)); print("[control] THINK_START")
                return "OK" if ok else "ERR"
            elif cmd=="THINK_STOP":
                ok=self._restore_idle(); print("[control] THINK_STOP")
                return "OK" if ok else "ERR"
            elif cmd=="TTS_START":
                ok=self._set_mode_face("TTS", _norm(TTS_FACE)); print("[control] TTS_START")
                return "OK" if ok else "ERR"
            elif cmd=="TTS_STOP":
                ok=self._restore_idle(); print("[control] TTS_STOP")
                return "OK" if ok else "ERR"
            elif cmd=="ERROR_START":
                ok=self._set_mode_face("ERROR", _norm(ERROR_FACE)); print("[control] ERROR_START")
                return "OK" if ok else "ERR"
            elif cmd=="ERROR_STOP":
                ok=self._restore_idle(); print("[control] ERROR_STOP")
                return "OK" if ok else "ERR"
            elif cmd=="SHOW_IDLE":
                ok=self._restore_idle(); print("[control] SHOW_IDLE")
                return "OK" if ok else "ERR"
            elif cmd=="FACE":
                if not args: return "ERR FACE needs <name>"
                ok=set_face(" ".join(args)); return "OK" if ok else "ERR"
            elif cmd=="BRI":
                if not args: return "ERR BRI needs <0..255>"
                try: lvl=max(0,min(255,int(args[0])))
                except: return "ERR bad BRI"
                ok=set_bri(lvl); return "OK" if ok else "ERR"
            elif cmd=="CLR":
                ok=screen_clear(); return "OK" if ok else "ERR"
            elif cmd=="MODE":
                return f"OK {self.mode}"
            else: return f"ERR unknown '{cmd}'"
        except Exception as e:
            return f"ERR {e}"

def _standalone():
    srv=ControlServer(sock_path=SOCK_PATH, on_rec_start=None, on_rec_stop=None)
    def _sh(*_):
        print("\n[control] shutting down..."); srv.stop_server(); sys.exit(0)
    signal.signal(signal.SIGINT,_sh); signal.signal(signal.SIGTERM,_sh)
    srv.start_server()
    while True: time.sleep(3600)

if __name__=="__main__":
    _standalone()
