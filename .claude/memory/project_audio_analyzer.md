---
name: Audio analyzer project
description: Python project for audio transcription, speaker diarization, sentiment analysis and summarization
type: project
---

Pipeline 4 étapes pour analyser des fichiers audio MP3/WAV :
1. Transcription + diarisation (resemblyzer + faster-whisper + sklearn AgglomerativeClustering)
2. Identification des locuteurs par empreintes vocales (SQLite)
3. Analyse de sentiment via Ollama (local LLM)
4. Résumé structuré via Ollama

**Why:** Outil d'analyse audio local, tout tourne en local (Whisper + Ollama), pas de dépendance cloud.

**How to apply:** Suggérer des solutions qui restent dans l'écosystème local (pas d'API cloud pour la transcription ou le LLM). Préférer des optimisations compatibles Windows + venv.

Stack clé : faster-whisper, resemblyzer, sklearn, sqlite3, ollama, librosa, webrtcvad-wheels (pas webrtcvad).

Point d'entrée : `main.py` — commandes CLI : analyze, add-fingerprint, list, show, fingerprints.
Config via `.env` (OLLAMA_HOST, OLLAMA_MODEL, WHISPER_MODEL_SIZE, AUDIO_LANGUAGE, DB_PATH).
