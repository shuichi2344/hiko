#!/usr/bin/env python3
# touch_bridge.py — line-based reader + face broker (single serial owner, locked)
import os, sys, time, socket, serial, threading

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK", "/tmp/hiko_serial.sock")

TOKENS = ("REC_START", "REC_STOP")

# One lock to rule them all: protect ALL reads/writes to 'ser'
SER_LOCK = threading.Lock()

def send_cmd_to_control(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            resp = s.recv(64).decode("utf-8", "ignore").strip()
            ok = resp.startswith("OK")
            print(f"[touch] -> {cmd} ({'OK' if ok else resp})", flush=True)
            return ok
    except Exception as e:
        print(f"[touch] control send error: {e}", flush=True)
        return False

def run_reader(ser: serial.Serial):
    # Short timeouts so we can keep looping even if nothing arrives
    ser.timeout = 0.2
    while True:
        try:
            with SER_LOCK:
                # read a whole line without blocking forever
                raw = ser.read_until(b"\n")
            line = raw.decode("utf-8", "ignore").strip()
        except Exception as e:
            print(f"[touch] read error: {e}", flush=True)
            raise
        if not line:
            continue
        print(f"[touch] raw: {line!r}", flush=True)
        if line in TOKENS:
            send_cmd_to_control(line)

def run_face_broker(ser: serial.Serial):
    try:
        if os.path.exists(SERIAL_SOCK):
            os.remove(SERIAL_SOCK)
    except Exception:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SERIAL_SOCK)
    os.chmod(SERIAL_SOCK, 0o666)
    srv.listen(8)
    print(f"[touch] face broker on {SERIAL_SOCK}", flush=True)

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_handle_face_client, args=(conn, ser), daemon=True).start()

def _handle_face_client(conn: socket.socket, ser: serial.Serial):
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
                    with SER_LOCK:
                        ser.write((cmd + "\n").encode("ascii", "ignore"))
                        # don't let flush block forever
                        try:
                            ser.flush()
                        except Exception:
                            pass
                    print(f"[face] wrote: {cmd}", flush=True)
                    # give the device a few ms to re-arm OUT before reader continues
                    time.sleep(0.01)
                    conn.sendall(b"OK\n")
                except Exception as e:
                    print(f"[face] write error: {e}", flush=True)
                    try:
                        conn.sendall(f"ERR {e}\n".encode("utf-8"))
                    except Exception:
                        pass
                    return

def _open_serial():
    # Try exclusive open; fall back if platform doesn't support it
    kw = dict(baudrate=BAUD, timeout=0.2, write_timeout=0.3)
    try:
        ser = serial.Serial(PORT, exclusive=True, **kw)  # Linux
    except TypeError:
        ser = serial.Serial(PORT, **kw)
    # Keep DTR asserted; RTS off. Some CDC stacks need DTR high to keep streaming.
    try:
        ser.dtr = True
        ser.rts = False
    except Exception:
        pass
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    return ser

def main():
    while True:
        try:
            print(f"[touch] opening {PORT} @ {BAUD}", flush=True)
            ser = _open_serial()
            print(f"[touch] listening on {PORT}", flush=True)

            # start broker thread (shares 'ser' via SER_LOCK)
            t_face = threading.Thread(target=run_face_broker, args=(ser,), daemon=True)
            t_face.start()

            # run reader (blocks until error)
            run_reader(ser)

        except (serial.SerialException, OSError) as e:
            print(f"[touch] serial/os error: {e}; retry in 1s", flush=True)
            time.sleep(1)
        finally:
            try: ser.close()
            except Exception: pass

if __name__ == "__main__":
    main()
