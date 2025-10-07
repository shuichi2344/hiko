# touch_bridge.py
import time, sys, serial, socket

PORT = "/dev/ttyACM0"
BAUD = 115200
CONTROL_SOCK = "/tmp/hiko_control.sock"

def send_cmd(cmd: str):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(CONTROL_SOCK)
            s.sendall((cmd.strip() + "\n").encode("utf-8"))
            s.recv(16)  # read "OK\n" (optional)
    except Exception as e:
        print(f"[touch] control send error: {e}", file=sys.stderr)

def main():
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=1) as ser:
                print(f"[touch] listening on {PORT} @ {BAUD}")
                while True:
                    line = ser.readline().decode("utf-8", "ignore").strip()
                    if not line:
                        continue
                    if line == "REC_START":
                        send_cmd("REC_START")
                    elif line == "REC_STOP":
                        send_cmd("REC_STOP")
        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)

if __name__ == "__main__":
    main()
