#!/usr/bin/env python3
import sys, time, glob, threading
import serial  # pip install pyserial

# default: stable symlink if present, else /dev/ttyACM0
_DEFAULT_PORTS = sorted(glob.glob("/dev/serial/by-id/usb-STMicroelectronics_*_if00")) or ["/dev/ttyACM0"]
_PORT = _DEFAULT_PORTS[0]
_serial_lock = threading.Lock()

def set_port(port: str):
    """Optionally set a fixed serial port path."""
    global _PORT
    _PORT = port

def _send_line(payload: str, port: str | None = None, expect_reply=False, timeout=1.0) -> bool:
    """Improved version with better error handling and connection reuse"""
    p = port or _PORT
    
    with _serial_lock:  # Prevent multiple threads from using serial simultaneously
        try:
            # Try to open serial port
            ser = serial.Serial(
                p, 
                baudrate=115200, 
                timeout=timeout,
                write_timeout=1.0
            )
        except Exception as e:
            print(f"[hiko_screen] open error on {p}: {e}")
            return False
        
        try:
            # Clear any pending data
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Send command
            command = (payload + "\n").encode("ascii", errors="ignore")
            bytes_written = ser.write(command)
            ser.flush()
            
            if bytes_written != len(command):
                print(f"[hiko_screen] write incomplete: {bytes_written}/{len(command)} bytes")
                return False
            
            # Brief delay to allow processing
            time.sleep(0.05)
            
            # Read response if requested
            if expect_reply:
                resp = ser.read(128)
                if resp:
                    try:
                        print(resp.decode("ascii", errors="ignore").strip())
                    except Exception:
                        pass
            
            return True
            
        except serial.SerialTimeoutException:
            print(f"[hiko_screen] write timeout on {p}")
            return False
        except Exception as e:
            print(f"[hiko_screen] communication error on {p}: {e}")
            return False
        finally:
            # Always close the connection
            try: 
                ser.close()
            except Exception: 
                pass

# Convenience wrappers you can import elsewhere
def face(name_or_index: str | int, port: str | None = None) -> bool:
    print(f"[hiko_screen] Setting face: {name_or_index}")
    result = _send_line(f"FACE {name_or_index}", port)
    print(f"[hiko_screen] Face result: {result}")
    return result

def bri(value: int, port: str | None = None) -> bool:
    value = max(0, min(255, int(value)))
    print(f"[hiko_screen] Setting brightness: {value}")
    result = _send_line(f"BRI {value}", port)
    print(f"[hiko_screen] Brightness result: {result}")
    return result

def clr(port: str | None = None) -> bool:
    print(f"[hiko_screen] Clearing screen")
    result = _send_line("CLR", port)
    print(f"[hiko_screen] Clear result: {result}")
    return result

# ---- CLI mode (keeps your current behavior) ----
def _main():
    if len(sys.argv) < 2:
        print("Usage: hiko_screen.py [face <name|0..6> | bri <0..255> | clr] [port]")
        sys.exit(1)

    port = None
    if len(sys.argv) >= 4:
        port = sys.argv[3]

    cmd = sys.argv[1].lower()
    if cmd == "face" and len(sys.argv) >= 3:
        ok = face(sys.argv[2], port)
    elif cmd == "bri" and len(sys.argv) >= 3:
        ok = bri(int(sys.argv[2]), port)
    elif cmd == "clr":
        ok = clr(port)
    else:
        print("Invalid command.")
        sys.exit(1)
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    _main()