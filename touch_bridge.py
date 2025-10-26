#!/usr/bin/env python3
# touch_bridge.py — broker: read REC_* from MCU, write face cmds to MCU; expose /tmp/hiko_serial.sock
import os, sys, time, socket, threading
import serial

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK",  "/tmp/hiko_serial.sock")
TOKENS = ("REC_START","REC_STOP")

def send_to_control(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.6)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd+"\n").encode())
            resp = s.recv(64).decode(errors="ignore").strip()
            return resp.startswith("OK")
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)
        return False

def face_socket_thread(ser: serial.Serial):
    # unix socket server for face commands
    try:
        if os.path.exists(SERIAL_SOCK):
            os.remove(SERIAL_SOCK)
    except Exception:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SERIAL_SOCK)
    os.chmod(SERIAL_SOCK, 0o666)
    srv.listen(8)
    print(f"[touch] screen command socket listening on {SERIAL_SOCK}")
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                data = b""
                try:
                    data = conn.recv(256)
                except Exception:
                    continue
                line = (data or b"").decode("utf-8","ignore").strip()
                if not line:
                    conn.sendall(b"ERR empty\n"); continue
                # Forward to MCU (append \n)
                try:
                    ser.write((line+"\n").encode("ascii","ignore"))
                    ser.flush()
                    conn.sendall(b"OK\n")
                except Exception as e:
                    conn.sendall(f"ERR {e}\n".encode())
    finally:
        try: srv.close()
        except: pass
        try: os.remove(SERIAL_SOCK)
        except: pass

def main():
    last_ok = None
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=0.02, inter_byte_timeout=0.02) as ser:
                print(f"[touch] broker up on {PORT} @ {BAUD} (exclusive owner)")
                try: ser.reset_input_buffer()
                except: pass

                # start face socket server
                t = threading.Thread(target=face_socket_thread, args=(ser,), daemon=True)
                t.start()

                buf = bytearray()
                while True:
                    chunk = ser.read(ser.in_waiting or 1)
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    s = buf.decode("utf-8","ignore")
                    i = 0
                    events = []
                    while True:
                        pos = min([p for p in (s.find("REC_START", i), s.find("REC_STOP", i)) if p != -1], default=-1)
                        if pos == -1: break
                        if s.startswith("REC_START", pos):
                            events.append("REC_START"); i = pos+9
                        else:
                            events.append("REC_STOP");  i = pos+8
                    if i: del buf[:i]
                    for ev in events:
                        if ev == last_ok:   # drop duplicate if last one ACKed
                            continue
                        if send_to_control(ev):
                            last_ok = ev
                            print(f"[touch] -> {ev}")
        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retry in 1s", file=sys.stderr)
            time.sleep(1)
            last_ok = None

if __name__ == "__main__":
    main()
