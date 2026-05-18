#!/bin/bash

# Installation script for OAAS Meeting Tools on Linux

echo "Installing OAAS Meeting Tools..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."

# Install webrtcvad-wheels (pre-compiled version for Linux)
pip install webrtcvad-wheels

# Install resemblyzer without dependencies to avoid compilation issues
pip install --no-deps "resemblyzer>=0.1.1.dev0"

# Install the rest of the dependencies
pip install \
    "faster-whisper>=1.0.0" \
    "ollama>=0.4.0" \
    "python-dotenv>=1.0.0" \
    "numpy>=1.24.0" \
    "torch>=2.0.0" \
    "torchaudio>=2.0.0" \
    "soundfile>=0.12.0" \
    "librosa>=0.10.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0"

echo "Installation complete!"
echo ""
echo "To activate the environment and use the tool:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the tool:"
echo "  python main.py analyze <fichier_audio> [--speakers N] [--threshold 0.75]"
echo ""
echo "Make sure Ollama is running locally:"
echo "  ollama serve"
echo "  ollama pull llama3"