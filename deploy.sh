#!/bin/bash

# Deployment script for OAAS Meeting Tools on Linux

echo "=== OAAS Meeting Tools Deployment ==="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.com"
    echo "After installation, run 'ollama pull llama3'"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify Ollama model is available
echo "Checking if llama3 model is available..."
if ! ollama list | grep -q "llama3"; then
    echo "llama3 model not found. Pulling it now..."
    ollama pull llama3
fi

echo "=== Deployment Complete ==="
echo "To use the tool:"
echo "  source .venv/bin/activate"
echo "  python main.py analyze <audio_file>"
echo ""
echo "Make sure Ollama is running:"
echo "  ollama serve"