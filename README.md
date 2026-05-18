# OAAS Meeting Tools

Audio Analysis tool for meeting transcription, speaker diarization, sentiment analysis, and summary generation - all running locally without requiring an account.

## Features

- Audio transcription with word-level timestamps
- Speaker diarization (identifying who spoke when)
- Speaker identification using voice fingerprints
- Sentiment analysis of each speaker's contributions
- Automated meeting summary generation
- Export to ICS format for Google Calendar
- Date detection from audio files and transcripts

## Requirements

- Python 3.8+
- Ollama (local LLM) for sentiment and summary analysis
- Linux system with audio support

## Installation

### Prerequisites

1. Install Ollama: https://ollama.com
2. Pull the required model:
   ```bash
   ollama pull llama3
   ```

### Install OAAS Meeting Tools

```bash
# Clone the repository
git clone <repository-url>
cd oaas-meeting-tools

# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

## Usage

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Basic Usage

```bash
# Analyze an audio file
python main.py analyze <fichier_audio> [--speakers N] [--threshold 0.75]

# Add a voice fingerprint for speaker identification
python main.py add-fingerprint "Nom" <fichier.wav>

# List all analyzed recordings
python main.py list

# Show details of a specific recording
python main.py show <id>

# Export recordings to ICS format
python main.py export-ics [<id> ...] [--output fichier.ics]
```

### Configuration

Create a `.env` file based on `.env.example` to customize settings:

```bash
cp .env.example .env
```

## Project Structure

```
oaas-meeting-tools/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── install.sh              # Installation script for Linux
├── .env.example            # Environment configuration example
├── CLAUDE.md               # Documentation for Claude AI
├── audio_analyzer/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── database.py         # SQLite database operations
│   ├── transcriber.py      # Audio transcription and diarization
│   ├── analyzer.py         # Sentiment analysis and summary generation
│   ├── fingerprint.py      # Voice fingerprint management
│   ├── date_detector.py    # Date detection from audio
│   └── ics_exporter.py     # ICS export functionality
```

## Development

### Setting up Development Environment

1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests or development commands as needed.

## License

This project is licensed under the MIT License.