#!/usr/bin/env python3
# touch_bridge.py — robust line reader + face broker (single serial owner)
import os, time, socket, serial, threading, queue

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK",  "/tmp/hiko_serial.sock")

TOKENS = ("REC_START", "REC_STOP")

# --- shared state ---
ser = None
stop_evt = threading.Event()
line_q: "queue.Queue[str]" = queue.Queue()
WRITE_LOCK = threading.Lock()   # serialize all writes to MCU

def send_cmd_to_control(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.6)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            resp = s.recv(64).decode("utf-8", "ignore").strip()
            ok = resp.startswith("OK")
            print(f"[touch] -> {cmd} ({'OK' if ok else resp})", flush=True)
            return ok
    except Exception as e:
        print(f"[touch] control send error: {e}", flush=True)
        return False

def open_serial():
    """Open serial exclusively and prepare low-latency reads."""
    global ser
    while not stop_evt.is_set():
        try:
            ser = serial.Serial(
                PORT, BAUD,
                timeout=0.05,              # short poll
                inter_byte_timeout=0.05,   # assemble fast
                exclusive=True             # we are the only owner
            )
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass
            # Some boards need a small settle after open
            time.sleep(0.05)
            print(f"[touch] serial open: {PORT} @ {BAUD}", flush=True)
            return
        except serial.SerialException as e:
            print(f"[touch] serial open failed: {e}; retrying…", flush=True)
            time.sleep(0.8)

def serial_reader():
    """Byte-buffered line assembly (no readline())."""
    buf = bytearray()
    while not stop_evt.is_set():
        try:
            # read whatever is available; at least 1 byte or timeout
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            # split on '\n'
            while True:
                nl = buf.find(b'\n')
                if nl < 0:
                    break
                raw, buf = buf[:nl], buf[nl+1:]
                s = raw.replace(b'\r', b'').decode('utf-8', 'ignore').strip()
                if s:
                    line_q.put(s)
        except serial.SerialException as e:
            print(f"[touch] serial read error: {e}", flush=True)
            try: ser.close()
            except: pass
            open_serial()
            buf.clear()
        except Exception as e:
            print(f"[touch] reader err: {e}", flush=True)

def event_pumper():
    """Forward REC_* lines to control; ignore other MCU chatter."""
    while not stop_evt.is_set():
        try:
            s = line_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if any(s.startswith(tok) for tok in TOKENS):
            send_cmd_to_control(s)
        # else: other logs are ignored (uncomment to debug)
        # print(f"[touch] MCU: {s}", flush=True)

def serial_write_line(cmd: str):
    """Thread-safe write to MCU (one line)."""
    if not cmd.endswith("\n"):
        cmd = cmd + "\n"
    with WRITE_LOCK:
        ser.write(cmd.encode("ascii", "ignore"))
        ser.flush()

def wait_reply_non_token(deadline: float) -> str | None:
    """Return first non-REC_* line before deadline (used for FACE/BRI/CLR optional acks)."""
    while time.monotonic() < deadline:
        try:
            s = line_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if any(s.startswith(tok) for tok in TOKENS):
            # requeue for pumper to see
            try: line_q.put_nowait(s)
            except: pass
            continue
        return s
    return None

def _handle_face_client(conn: socket.socket):
    with conn:
        buf = b""
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode("utf-8", "ignore").strip()
                if not cmd:
                    continue
                try:
                    serial_write_line(cmd)
                    print(f"[face] wrote: {cmd}", flush=True)
                    # Optional: wait briefly for any reply line that is NOT a REC_* token
                    reply = wait_reply_non_token(time.monotonic() + 0.30)
                    if reply:
                        conn.sendall(("OK " + reply + "\n").encode("utf-8"))
                    else:
                        conn.sendall(b"OK\n")
                except Exception as e:
                    print(f"[face] write error: {e}", flush=True)
                    try: conn.sendall(f"ERR {e}\n".encode("utf-8"))
                    except: pass
                    return

def run_face_broker():
    """UNIX socket that accepts FACE/BRI/CLR and forwards to MCU."""
    try:
        if os.path.exists(SERIAL_SOCK):
            os.remove(SERIAL_SOCK)
    except: pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SERIAL_SOCK)
    os.chmod(SERIAL_SOCK, 0o666)
    srv.listen(6)
    print(f"[touch] face broker on {SERIAL_SOCK}", flush=True)
    while not stop_evt.is_set():
        try:
            srv.settimeout(0.5)
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[touch] sock accept err: {e}", flush=True)
            continue
        threading.Thread(target=_handle_face_client, args=(conn,), daemon=True).start()

def main():
    print("[touch] starting bridge…", flush=True)
    open_serial()
    threading.Thread(target=serial_reader,  daemon=True).start()
    threading.Thread(target=event_pumper,   daemon=True).start()
    threading.Thread(target=run_face_broker,daemon=True).start()
    print("[touch] running.", flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        try: ser.close()
        except: pass

if __name__ == "__main__":
    main()
