#!/usr/bin/env python3
import os, socket, serial, threading, re, time

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("HIKO_SERIAL_BAUD", "115200"))
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK", "/tmp/hiko_serial.sock")

TOKEN_RE = re.compile(r"\b(REC[_ ]START|REC[_ ]STOP)\b")

def send_to_control(cmd: str, timeout=0.6) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode())
            resp = s.recv(64).decode("utf-8","ignore").strip()
            return resp.startswith("OK")
    except Exception:
        return False

def serve_serial_sock(ser: serial.Serial):
    # simple line-based UNIX socket server -> serial
    try:
        if os.path.exists(SERIAL_SOCK):
            os.remove(SERIAL_SOCK)
    except Exception:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SERIAL_SOCK)
    os.chmod(SERIAL_SOCK, 0o666)
    srv.listen(8)
    print(f"[bridge] serial sock up at {SERIAL_SOCK}")

    def handle(conn):
        with conn:
            buf = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode("utf-8","ignore").strip()
                    if not cmd:
                        continue
                    try:
                        ser.write((cmd + "\n").encode("ascii","ignore"))
                        ser.flush()
                        conn.sendall(b"OK\n")
                    except Exception:
                        try: conn.sendall(b"ERR\n")
                        except Exception: pass
                        return

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()

def main():
    with serial.Serial(PORT, BAUD, timeout=1, rtscts=False, dsrdtr=False) as ser:
        print(f"[bridge] connected to {PORT} @ {BAUD} (combined)")
        threading.Thread(target=serve_serial_sock, args=(ser,), daemon=True).start()

        last_evt = None
        while True:
            try:
                line = ser.readline().decode("utf-8","ignore").strip()
            except Exception as e:
                print(f"[bridge] read error: {e}")
                time.sleep(0.5)
                continue
            if not line:
                continue
            m = TOKEN_RE.search(line)
            if not m:
                # uncomment to debug other lines: print("[bridge] passthru:", line)
                continue
            evt = "REC_START" if "START" in m.group(1) else "REC_STOP"
            if evt == last_evt:
                continue
            if send_to_control(evt):
                last_evt = evt
                print(f"[bridge] -> {evt}")
            else:
                print(f"[bridge] control send failed: {evt}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
