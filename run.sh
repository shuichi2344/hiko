#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ensure venv
source hiko-env/bin/activate

# run the voice loop (change --device if needed)
python scripts/demo_voice_loop.py --device 1
