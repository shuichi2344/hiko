# Hiko – The AI Desktop Pet

Hiko is an interactive **AI-powered desktop companion** built on the Raspberry Pi 5.  
It listens, thinks, and speaks — combining **wake-word detection**, **speech recognition**, **local AI (LLM)**, and **text-to-speech** into one compact assistant.

---

## Key Features

- **Voice Interaction**  
  - Hotword detection (Porcupine)  
  - Microphone input via **ReSpeaker 2-Mic HAT**  
- **Speech-to-Text (STT)**  
  - Local transcription with **Whisper.cpp**  
- **AI Brain**  
  - Runs **Ollama** locally for lightweight LLM responses  
- **Text-to-Speech (TTS)**  
  - Natural speech output with **Piper TTS**  
- **Runs Offline**  
  - Works fully on-device — no constant cloud connection needed  

---

## Hardware Requirements

- Raspberry Pi 5 (4GB or 8GB recommended)  
- microSD card (32GB+)  
- Power supply (at least 5V 5A for Pi 5)  
- ReSpeaker 2-Mic HAT (for microphone input)  
- Speakers or headphones (3.5mm jack or HDMI)  
- Optional: USB mic (if you don't want to use the HAT)  

---

## Software Requirements

- Raspberry Pi OS **Bookworm** (Debian 12)  
- Python 3.11+  
- Git, CMake, Build Essentials  
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)  
- [Ollama](https://ollama.ai)  
- [Piper TTS](https://github.com/rhasspy/piper)  
- Porcupine SDK (for wake word detection)  

---

## Setup Guide

### 1. System Preparation

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git build-essential cmake \
  python3 python3-venv python3-pip \
  portaudio19-dev python3-pyaudio sox \
  minicom screen
```

### 2. Clone this Repo

```bash
git clone https://github.com/shuichi2344/hiko.git
cd hiko
```

### 3. Virtual Environment

```bash
python3 -m venv hiko-env
source hiko-env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. ReSpeaker 2-Mic HAT Setup (Pi 5, Bookworm)

The 2-Mic HAT overlay is not preinstalled on Pi 5. You must build it manually:

```bash
# Install compiler
sudo apt install -y device-tree-compiler

# Get overlay sources
cd ~
git clone https://github.com/Seeed-Studio/seeed-linux-dtoverlays.git
cd seeed-linux-dtoverlays/overlays/rpi

# Build & install overlay
dtc -@ -I dts -O dtb -o respeaker-2mic-v2_0-overlay.dtbo respeaker-2mic-v2_0-overlay.dts
sudo cp respeaker-2mic-v2_0-overlay.dtbo /boot/firmware/overlays/

# Enable overlay
sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.bak.$(date +%F)
printf "\n# ReSpeaker 2-Mic HAT\ndtparam=audio=off\ndtoverlay=respeaker-2mic-v2_0-overlay\n" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

**Verify mic is detected:**
```bash
arecord -l
# Expect: card 0: seeed2micvoicec [...]
```

**Adjust volume:**
```bash
alsamixer
# F6 → select seeed2micvoicec
# Recommended levels:
#   PCM: ~75%
#   Playback: ~75%
#   Line DAC: ~70%
#   Capture: ~70–75%
#   Mic Boost: off (0 dB) or +6 dB max
sudo alsactl store
```

### 5. Whisper.cpp (Speech-to-Text) - English

```bash
cd ~/hiko
git submodule add https://github.com/ggerganov/whisper.cpp

# Build whisper.cpp (CMake)
cd whisper.cpp
cmake -B build
cmake --build build -j4
cd ..

# Download English-only model (better accuracy for EN)
bash whisper.cpp/models/download-ggml-model.sh small.en

# Quantize for speed on Pi 5:
cmake --build whisper.cpp/build -j4 --target quantize
whisper.cpp/build/bin/quantize \
  whisper.cpp/models/ggml-small.en.bin \
  whisper.cpp/models/ggml-small.en-q5_1.bin q5_1
```

### 6. Ollama (Local LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:1b
```

### 7. Piper (Text-to-Speech)

Download a voice model (example):

```bash
# Install Piper CLI (inside your venv)
source hiko-env/bin/activate
pip install piper-tts

mkdir -p piper-voices
# Place .onnx + .json voice model files here
```

### 8. Test Microphone

```bash
# List devices and find your ReSpeaker (card `seeed2micvoicec`):
arecord -l

# Testing
source hiko-env/bin/activate
python scripts/mic_test.py --seconds 5 --device 1 --outfile test.wav
aplay test.wav
```

### 9. Voice Demo (record → transcribe → LLM → TTS)

```bash
source hiko-env/bin/activate
python scripts/demo_voice_loop.py --device 1
```

## How to Run

```bash
# 1) Start Ollama (once)
ollama serve &

# 2) Activate venv
source hiko-env/bin/activate

# 3) Voice demo (record → transcribe → LLM → TTS)
python scripts/demo_voice_loop.py --device 1
```

---

## Project Structure

```
hiko/
├── hiko-env/              # Python virtual environment (ignored in Git)
├── scripts/               # Project Python scripts
│   ├── demo_voice_loop.py # Main voice loop (STT → LLM → TTS)
│   ├── demo_voice_loop.py.bak # Backup of voice loop (not tracked normally)
│   └── mic_test.py        # Test ReSpeaker microphone
├── whisper.cpp/           # Speech-to-text engine (submodule)
│   ├── build/             # Build artifacts (ignored in Git)
│   └── models/            # Whisper models (.bin/.gguf, ignored in Git)
├── piper-voices/          # Piper TTS models (.onnx, ignored in Git)
├── out/                   # Audio recordings & transcripts (ignored in Git)
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment setup script
├── run.sh                 # Main run script (launches demo)
└── README.md              # Project documentation

```

---

## Quick Start

1. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Test your microphone:**
   ```bash
   source hiko-env/bin/activate
   python scripts/mic_test.py --seconds 3 --device 0
   ```

3. **Start Hiko:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

---

## Configuration

### Audio Device Selection

Use the mic test script to find the correct device index:

```bash
python scripts/mic_test.py --device 1  # Try different indices
```

### Voice Model Selection

Place your preferred Piper TTS voice models in the `piper-voices/` directory. Each model consists of:
- `.onnx` file (the neural network model)
- `.json` file (voice configuration)

### Wake Word Customization

Configure your preferred wake word in the Porcupine settings. 

---

## Troubleshooting

### Microphone Issues

1. **No audio input detected:**
   ```bash
   arecord -l  # List all audio devices
   alsamixer   # Check volume levels
   ```

2. **Poor audio quality:**
   - Adjust microphone gain in `alsamixer`
   - Check for background noise
   - Ensure proper HAT installation

### Performance Issues

1. **Slow response times:**
   - Use the quantized Whisper model: `ggml-small.en-q5_1.bin`
   - Use lighter LLM models (llama3.2:1b, phi3:mini, qwen2.5:1.5b-instruct)
   - Close unnecessary background processes

2. **High CPU usage:**
   - Monitor with `htop`
   - Consider using hardware acceleration if available
   - Lower LLM context length / threads:
     ```bash
     export OLLAMA_NUM_THREADS=4
     export OLLAMA_NUM_PARALLEL=1
     export OLLAMA_CONTEXT_LENGTH=2048
     ```

### Common Errors

- **"Device not found"**: Check device index in mic test
- **"Permission denied"**: Ensure user is in `audio` group
- **"Model not found"**: Verify model files are in correct directories

---

## Dependencies

### Python Packages
- `pvporcupine` - Wake word detection
- `pvrecorder` - Audio recording
- `pyaudio` - Audio I/O
- `sounddevice` - Alternative audio interface
- `pyserial` - Serial communication
- `piper-tts` - Text-to-speech synthesis

### System Dependencies
- `portaudio19-dev` - Audio library development files
- `python3-pyaudio` - Python audio bindings
- `sox` - Audio processing tools
- `build-essential` - Compilation tools
- `cmake` - Build system

---

## Acknowledgments

- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for local speech recognition
- [Ollama](https://ollama.ai) for local LLM inference
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech
- [Porcupine](https://picovoice.ai/products/porcupine/) for wake word detection
- [Seeed Studio](https://www.seeedstudio.com/) for the ReSpeaker 2-Mic HAT

---
