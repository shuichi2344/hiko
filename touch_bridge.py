#!/usr/bin/env python3
# touch_bridge.py — single owner of STM32 CDC, forwards REC_* to control socket,
# and exposes a UNIX socket to accept FACE/BRI/CLR commands (no re-open).

import os, sys, time, socket, threading
import serial

# ---- Env
PORT         = os.environ.get("HIKO_SERIAL_PORT", "/dev/ttyACM0")
BAUD         = int(os.environ.get("HIKO_SERIAL_BAUD", "115200"))
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")
SERIAL_SOCK  = os.environ.get("HIKO_SERIAL_SOCK",  "/tmp/hiko_serial.sock")

TOKENS = ("REC_START", "REC_STOP")

def _send_control(cmd: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.6)
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip()+"\n").encode("utf-8"))
            resp = s.recv(64).decode("utf-8","ignore").strip()
            return resp.startswith("OK")
    except Exception as e:
        print(f"[bridge] control send error: {e}", file=sys.stderr)
        return False

def _extract_events(buf: bytearray):
    out = []
    s = buf.decode("utf-8", "ignore")
    i = 0
    while True:
        pos = min([p for p in (s.find("REC_START", i), s.find("REC_STOP", i)) if p != -1], default=-1)
        if pos < 0: break
        tok = "REC_START" if s.startswith("REC_START", pos) else "REC_STOP"
        out.append(tok)
        i = pos + len(tok)
    if i:
        del buf[:i]
    if len(buf) > 2048:
        del buf[:-256]
    return out

def _socket_server(ser: serial.Serial):
    # Creates /tmp/hiko_serial.sock, accepts lines like:
    #   FACE happy
    #   BRI 180
    #   CLR
    try:
        if os.path.exists(SERIAL_SOCK):
            os.remove(SERIAL_SOCK)
    except Exception:
        pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SERIAL_SOCK)
    os.chmod(SERIAL_SOCK, 0o666)
    srv.listen(4)
    print(f"[bridge] screen command socket up on {SERIAL_SOCK}")

    while True:
        try:
            conn, _ = srv.accept()
        except Exception:
            continue
        threading.Thread(target=_handle_client, args=(conn, ser), daemon=True).start()

def _handle_client(conn: socket.socket, ser: serial.Serial):
    with conn:
        try:
            data = conn.recv(256)
        except Exception:
            return
        if not data:
            return
        line = data.decode("utf-8","ignore").strip()
        ok = _handle_screen_cmd(line, ser)
        try:
            conn.sendall(b"OK\n" if ok else b"ERR\n")
        except Exception:
            pass

def _handle_screen_cmd(line: str, ser: serial.Serial) -> bool:
    # Pass straight through to MCU as a single newline-terminated line
    if not line:
        return False
    try:
        ser.write((line.strip()+"\n").encode("ascii", "ignore"))
        ser.flush()
        return True
    except Exception as e:
        print(f"[bridge] write error: {e}", file=sys.stderr)
        return False

def main():
    last_ok = None
    print(f"[bridge] opening {PORT} @ {BAUD}")
    while True:
        try:
            with serial.Serial(
                PORT, BAUD,
                timeout=0.02,
                inter_byte_timeout=0.02,
                exclusive=True,           # single owner!
                rtscts=False, dsrdtr=False, xonxoff=False
            ) as ser:
                # Keep DTR asserted if your STM32 expects it, and leave RTS low.
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

                # Start socket server for FACE/BRI/CLR (once per open)
                threading.Thread(target=_socket_server, args=(ser,), daemon=True).start()
                print("[bridge] listening for tokens and screen commands...")

                buf = bytearray()
                while True:
                    chunk = ser.read(ser.in_waiting or 1)
                    if chunk:
                        buf.extend(chunk)
                        for evt in _extract_events(buf):
                            if evt == last_ok:
                                continue
                            if _send_control(evt):
                                last_ok = evt
                                print(f"[bridge] -> {evt}")

        except serial.SerialException as e:
            print(f"[bridge] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)
            last_ok = None

if __name__ == "__main__":
    main()
