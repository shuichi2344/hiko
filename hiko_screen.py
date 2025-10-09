#!/usr/bin/env python3
import sys, time
import serial

# Usage:
#   python hiko_screen.py face happy
#   python hiko_screen.py face 1
#   python hiko_screen.py bri 200
#   python hiko_screen.py clr
#
# If your port isn’t /dev/ttyACM0, pass it as the last arg:
#   python hiko_screen.py face happy /dev/serial/by-id/usb-ST...

def main():
    if len(sys.argv) < 2:
        print("Usage: hiko_screen.py [face <name|0..6> | bri <0..255> | clr] [port]")
        sys.exit(1)

    # default port
    port = "/dev/ttyACM0"
    if len(sys.argv) >= 4:
        port = sys.argv[3]

    cmd = sys.argv[1].lower()
    if cmd == "face" and len(sys.argv) >= 3:
        payload = f"FACE {sys.argv[2]}"
    elif cmd == "bri" and len(sys.argv) >= 3:
        payload = f"BRI {sys.argv[2]}"
    elif cmd == "clr":
        payload = "CLR"
    else:
        print("Invalid command.")
        sys.exit(1)

    ser = serial.Serial(port, baudrate=115200, timeout=0.5)  # baud is ignored by CDC, but fine
    ser.write((payload + "\n").encode("ascii"))
    ser.flush()
    time.sleep(0.05)
    ser.close()

if __name__ == "__main__":
    main()
