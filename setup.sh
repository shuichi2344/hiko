#!/usr/bin/env bash
set -e
sudo apt update && sudo apt install -y \
  build-essential cmake python3 python3-venv python3-pip \
  portaudio19-dev python3-pyaudio sox git

python3 -m venv hiko-env
source hiko-env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Build whisper.cpp if present
if [ -d whisper.cpp ]; then
  (cd whisper.cpp && make)
fi
echo "Setup complete."
