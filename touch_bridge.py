#!/usr/bin/env python3
import os, time, sys, socket, select, serial, threading, queue

PORT = os.environ.get("HIKO_SERIAL_PORT", "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_6D7433A15248-if00")
BAUD = 115200
CONTROL_SOCK = os.environ.get("HIKO_CONTROL_SOCK", "/tmp/hiko_control.sock")

# Send to control server quickly, but don't block the reader.
outq = queue.Queue(maxsize=128)

def sender():
    while True:
        msg = outq.get()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                s.connect(CONTROL_SOCK)
                s.sendall((msg.strip() + "\n").encode("utf-8"))
                # Non-essential: best-effort read; ignore if controller busy
                try: s.recv(64)
                except: pass
        except Exception:
            pass  # controller might be busy; we'll try next event

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0)  # non-blocking
    ser.reset_input_buffer()
    buf = bytearray()

    # Start sender thread
    threading.Thread(target=sender, daemon=True).start()

    poll = select.poll()
    poll.register(ser.fileno(), select.POLLIN)

    last_emit = 0.0
    debounce_ms = 120   # deglitch: ignore events within this window

    while True:
        # Wait up to 50ms for serial, so CPU stays cool
        ev = poll.poll(50)
        if ev:
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)
                while True:
                    nl = buf.find(b'\n')
                    if nl < 0: break
                    line = buf[:nl].decode('utf-8', 'ignore').strip()
                    del buf[:nl+1]
                    now = time.time()*1000
                    if now - last_emit >= debounce_ms and line in ("REC_START","REC_STOP"):
                        last_emit = now
                        try: outq.put_nowait(line)
                        except queue.Full: pass
                        # Optional: log to stdout for your test shell
                        print(time.strftime('[%H:%M:%S]'), line, flush=True)
        # Tiny sleep to yield if nothing to do
        time.sleep(0.005)

if __name__ == "__main__":
    main()
