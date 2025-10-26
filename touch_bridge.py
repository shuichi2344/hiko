#!/usr/bin/env python3
import os, socket, serial, threading, re, time, queue

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("HIKO_SERIAL_BAUD", "115200"))
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK",  "/tmp/hiko_serial.sock")

# TX behavior (tuneable via envs)
IDLE_MS      = int(os.environ.get("FACE_TX_IDLE_MS", "120"))   # wait this long with no RX before sending
MIN_TX_GAPMS = int(os.environ.get("FACE_TX_GAP_MS", "80"))     # min gap between consecutive TX lines
LINE_ENDING  = os.environ.get("FACE_TX_EOL", "\\n").encode().decode("unicode_escape")  # "\n" or "\r\n"

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

def serve_serial_sock(ser: serial.Serial, txq: "queue.Queue[str]", last_face_ref: dict):
    # UNIX socket that accepts FACE/BRI/CLR and enqueues them
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
                    # dedupe faces to cut chatter
                    if cmd.startswith("FACE "):
                        if last_face_ref.get("face") == cmd:
                            try: conn.sendall(b"OK\n")
                            except Exception: pass
                            continue
                        last_face_ref["face"] = cmd
                    try:
                        txq.put_nowait(cmd)
                        conn.sendall(b"OK\n")
                    except Exception:
                        try: conn.sendall(b"ERR\n")
                        except Exception: pass
                        return

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()

def writer_thread(ser: serial.Serial, txq: "queue.Queue[str]", rx_state: dict):
    last_tx_ts = 0.0
    while True:
        cmd = txq.get()  # blocks
        # wait for RX idle
        while True:
            now = time.time()
            idle_ok = (now - rx_state.get("last_rx", 0.0)) * 1000.0 >= IDLE_MS
            there_is_input = ser.in_waiting > 0
            gap_ok = (now - last_tx_ts) * 1000.0 >= MIN_TX_GAPMS
            if idle_ok and not there_is_input and gap_ok:
                break
            time.sleep(0.01)

        # transmit one line
        payload = (cmd + LINE_ENDING).encode("ascii","ignore")
        try:
            ser.write(payload)
            ser.flush()
            last_tx_ts = time.time()
            # tiny settle delay
            time.sleep(MIN_TX_GAPMS / 1000.0)
        except Exception as e:
            print(f"[bridge] TX error: {e}")
        finally:
            txq.task_done()

def main():
    # Open serial in the safest posture for your STM32
    with serial.Serial(PORT, BAUD, timeout=1, rtscts=False, dsrdtr=False) as ser:
        # Explicitly hold DTR True, RTS False like your working reader
        try:
            ser.dtr = True
            ser.rts = False
        except Exception:
            pass

        print(f"[bridge] connected to {PORT} @ {BAUD} (combined; polite TX)")
        txq = queue.Queue(maxsize=128)
        rx_state = {"last_rx": time.time()}
        last_face_ref = {"face": None}

        # start UNIX socket server for face commands
        threading.Thread(target=serve_serial_sock, args=(ser, txq, last_face_ref), daemon=True).start()
        # start writer thread
        threading.Thread(target=writer_thread, args=(ser, txq, rx_state), daemon=True).start()

        last_evt = None
        # RX loop (line-based)
        while True:
            try:
                chunk = ser.readline()  # mirrors your working code
            except Exception as e:
                print(f"[bridge] read error: {e}")
                time.sleep(0.5)
                continue

            if not chunk:
                continue

            rx_state["last_rx"] = time.time()
            line = chunk.decode("utf-8","ignore").strip()
            m = TOKEN_RE.search(line)
            if not m:
                # uncomment to see everything: print("[bridge] rx:", line)
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
