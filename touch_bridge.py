# touch_bridge.py
import time, sys, serial

# Use your device path; /dev/ttyACM0 is common. We'll make it stable in Step 5.
PORT = "/dev/ttyACM0"
BAUD = 115200

# ===== Hook into chatbot.py =====
# Option A (best): import real functions from your chatbot
try:
    from chatbot import start_recording, stop_recording
except Exception as e:
    print(f"[touch] Warning: couldn’t import chatbot hooks: {e}")
    # Fallback stubs for quick testing:
    def start_recording(): print("[touch] -> start_recording()")
    def stop_recording():  print("[touch] -> stop_recording()")

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
                        start_recording()
                    elif line == "REC_STOP":
                        stop_recording()
        except serial.SerialException as e:
            print(f"[touch] serial error: {e}; retrying in 1s", file=sys.stderr)
            time.sleep(1)

if __name__ == "__main__":
    main()
