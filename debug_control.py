# debug_control.py
#!/usr/bin/env python3
import os, sys, socket, threading, time, traceback
from hiko_screen import face as set_face, bri as set_bri, clr as screen_clear

SOCK_PATH = "/tmp/hiko_control.sock"

class DebugControlServer:
    def __init__(self):
        self.sock_path = SOCK_PATH
        self._stop = threading.Event()
        
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
        print(f"[debug] listening on {self.sock_path}")
        self._srv.settimeout(1.0)
        
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[debug] accept error: {e}")
                break
    
    def _handle_client(self, conn):
        with conn:
            try:
                data = conn.recv(1024).decode('utf-8').strip()
                print(f"[debug] received: {data}")
                
                response = self._process_command(data)
                print(f"[debug] response: {response}")
                
                conn.sendall((response + "\n").encode('utf-8'))
            except Exception as e:
                print(f"[debug] client error: {e}")
                conn.sendall(b"ERROR client handling failed\n")
    
    def _process_command(self, command):
        parts = command.split()
        if not parts:
            return "ERR empty command"
        
        cmd = parts[0].upper()
        args = parts[1:]
        
        print(f"[debug] processing: {cmd} with args {args}")
        
        if cmd == "PING":
            return f"OK PONG {int(time.time())}"
        
        elif cmd == "FACE":
            if not args:
                return "ERR FACE needs argument"
            face_name = " ".join(args)
            print(f"[debug] Attempting to set face: {face_name}")
            try:
                result = set_face(face_name)
                print(f"[debug] set_face result: {result}")
                return "OK" if result else "ERR face failed"
            except Exception as e:
                print(f"[debug] set_face exception: {e}")
                return f"ERR face exception: {e}"
        
        elif cmd == "REC_START":
            print("[debug] REC_START command")
            try:
                result = set_face("shy")
                print(f"[debug] REC_START face result: {result}")
                return "OK" if result else "ERR REC_START face failed"
            except Exception as e:
                print(f"[debug] REC_START exception: {e}")
                return f"ERR REC_START exception: {e}"
        
        elif cmd == "REC_STOP":
            print("[debug] REC_STOP command")
            try:
                result = set_face("happy")
                print(f"[debug] REC_STOP face result: {result}")
                return "OK" if result else "ERR REC_STOP face failed"
            except Exception as e:
                print(f"[debug] REC_STOP exception: {e}")
                return f"ERR REC_STOP exception: {e}"
        
        else:
            return f"ERR unknown command: {cmd}"
    
    def stop_server(self):
        self._stop.set()
        try:
            self._srv.close()
        except Exception:
            pass
        try:
            os.remove(self.sock_path)
        except Exception:
            pass

if __name__ == "__main__":
    server = DebugControlServer()
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n[debug] Shutting down...")
        server.stop_server()