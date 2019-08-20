#!/bin/sh
set -x
python3 examples/vad_transcriber/audioTranscript_cmd.py --model models --stream
