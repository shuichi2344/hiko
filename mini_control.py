#!/usr/bin/env python3
import os, socket, signal, sys
# from hiko_screen import face as set_face, bri as set_bri, clr as screen_clear, set_port as screen_set_port
from hiko_screen_shim import face as set_face, bri as set_bri, clr as screen_clear, set_port as screen_set_port

SOCK="/tmp/hiko_control.sock"
PORT=os.environ.get("HIKO_SERIAL_PORT",
    "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_Port_6D7433A15248-if00")
print("[mini] starting...", flush=True)
print("HIKO_SERIAL_PORT =", os.environ.get("HIKO_SERIAL_PORT"), flush=True)

def main():
    try: os.remove(SOCK)
    except: pass
    set_port(PORT)
    ok=set_face(os.environ.get("HIKO_IDLE_FACE","neutral"))
    print(f"[mini] listening on {SOCK}; idle set -> {ok}")
    s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK); os.chmod(SOCK,0o666); s.listen(8)
    def bye(*_): 
      try: s.close(); os.remove(SOCK)
      except: pass
      sys.exit(0)
    signal.signal(signal.SIGINT, bye); signal.signal(signal.SIGTERM, bye)
    while True:
        c,_=s.accept()
        line=c.recv(1024).decode().strip()
        if line.upper().startswith("FACE "):
            name=line.split(" ",1)[1]
            rep="OK" if set_face(name) else "ERR face failed"
        elif line.upper()=="PING":
            rep="OK PONG"
        else:
            rep="OK"
        c.sendall((rep+"\n").encode()); c.close()  # <- always close so client prints

if __name__=="__main__": main()
