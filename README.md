# OAAS Meeting Tools

Audio Analysis tool for meeting transcription, speaker diarization, sentiment analysis, and summary generation - all running locally without requiring an account.

## Features

- Audio transcription with word-level timestamps
- Speaker diarization (identifying who spoke when)
- Speaker identification using voice fingerprints
- Sentiment analysis of each speaker's contributions
- Automated meeting summary generation
- Pattern detection: availability (busy), sleeping over, audio quality issues
- Transcript search with keyword profiles and Ollama confirmation
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
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\Activate.ps1
```

### Commands

#### Analysis

```bash
# Analyze a single audio file
python main.py analyze <fichier.mp3> [--speakers N] [--threshold 0.75]

# With pattern detection for a known speaker (fingerprint ID from `fingerprints` command)
# Detects: busy/unavailable and sleeping-over for that speaker, audio quality for the whole recording
python main.py analyze <fichier.mp3> --profile-fingerprint <FP_ID>

# Scan an entire directory and analyze all audio files found
# Supported formats: mp3, wav, m4a, ogg, flac, opus, aac
# Already-analyzed files are automatically skipped
python main.py scan-dir <répertoire> [--speakers N] [--threshold 0.75]
python main.py scan-dir <répertoire> --recursive              # include subdirectories
python main.py scan-dir <répertoire> --profile-fingerprint <FP_ID>
```

#### Speakers & Fingerprints

```bash
# Add a voice fingerprint for a known speaker
python main.py add-fingerprint "Nom" <fichier.wav>

# List registered voice fingerprints (shows sample count and custom threshold)
python main.py fingerprints

# Remove a fingerprint
python main.py remove-fingerprint <id>

# Improve detection for a speaker who speaks in short clips:

# 1. Extract all their audio segments from existing recordings into a WAV file
python main.py extract-speaker-audio <fp_id> [--output speaker.wav] [--min-duration 0.5]

# 2. Enrich the fingerprint by averaging in the new clip (weighted by sample count)
python main.py enrich-fingerprint <fp_id> <fichier.wav>

# 3. Optionally set a lower detection threshold for this speaker only
python main.py set-fingerprint-threshold <fp_id> 0.60
python main.py set-fingerprint-threshold <fp_id>        # reset to global threshold
```

#### Viewing Results

```bash
# List all analyzed recordings
python main.py list

# Show details of a specific recording (transcript, speakers, detections, summary)
python main.py show <id> [--no-transcript]
```

#### Search

```bash
# Free-text search across all transcripts (with Ollama confirmation)
python main.py search "occupé"

# Predefined profiles: busy | sleeping | quality
python main.py search --profile busy
python main.py search --profile sleeping
python main.py search --profile quality

# Scope to a specific recording
python main.py search --profile quality --recording <id>

# Filter by speaker name
python main.py search --profile busy --speaker "Alice"

# Skip Ollama confirmation (keyword-only, instant)
python main.py search "ça coupe" --no-confirm
```

Profiles scan for these patterns:
| Profile | Detects |
|---------|---------|
| `busy` | "occupé", "pas disponible", "j'ai pas le temps", "je peux pas"… |
| `sleeping` | "dormir chez", "passer la nuit", "je reste chez"… |
| `quality` | "ça coupe", "tu m'entends", "j'entends pas", "connexion"… |

#### Export & Maintenance

```bash
# Export all recordings to CSV (id, date, recognized speakers, transcription, summary)
python main.py export-csv
python main.py export-csv --output rapport.csv

# Export recordings to ICS format for Google Calendar
python main.py export-ics [<id> ...] [--output fichier.ics]

# Backfill missing dates on existing recordings
python main.py backfill-dates

# Run a profile detection on all existing recordings (skips already-processed ones)
python main.py backfill-detections --profile quality
python main.py backfill-detections --profile busy
python main.py backfill-detections --profile sleeping

# Scope busy/sleeping detection to a specific identified speaker
python main.py backfill-detections --profile busy --fingerprint <FP_ID>
python main.py backfill-detections --profile sleeping --fingerprint <FP_ID>

# Delete ALL recordings from the database (voice fingerprints are kept)
python main.py clear-recordings
python main.py clear-recordings --yes   # skip confirmation prompt
```

> Ollama must be running before any analysis: `ollama serve`

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
│   ├── searcher.py         # Transcript search with keyword profiles and Ollama confirmation
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