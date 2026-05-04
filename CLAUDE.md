# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commandes courantes

```powershell
# Activer le venv (Windows)
.venv\Scripts\Activate.ps1

# Installer les dépendances (Windows, sans Visual C++)
.\install.ps1
# ou manuellement :
.venv\Scripts\pip install -r requirements.txt

# Lancer une analyse complète
.venv\Scripts\python main.py analyze <fichier.mp3> [--speakers N] [--threshold 0.75]

# Enregistrer une empreinte vocale (locuteur connu)
.venv\Scripts\python main.py add-fingerprint "Nom" <fichier.wav>

# Consulter les résultats
.venv\Scripts\python main.py list
.venv\Scripts\python main.py show <id> [--no-transcript]
.venv\Scripts\python main.py fingerprints
```

Ollama doit tourner localement avant toute analyse : `ollama serve` + le modèle configuré dans `.env` doit être pulled (`ollama pull llama3`).

## Architecture

Pipeline en 4 étapes orchestré dans `main.py:cmd_analyze` :

1. **Transcription + diarisation** (`audio_analyzer/transcriber.py`)
   - `librosa.effects.split` détecte les segments de parole (VAD énergie, sans modèle externe)
   - `resemblyzer.VoiceEncoder` extrait un embedding 256-dim par segment
   - `sklearn.AgglomerativeClustering` (cosine, distance_threshold=0.35) regroupe les segments par locuteur
   - `faster-whisper` transcrit l'audio avec timestamps mot par mot
   - `_dominant_speaker` fusionne les deux en assignant à chaque segment Whisper le locuteur avec le plus grand overlap temporel

2. **Identification des locuteurs** (`audio_analyzer/fingerprint.py`)
   - Optionnel : si des empreintes nommées existent en base, chaque locuteur détecté est comparé par similarité cosinus
   - Seuil configurable via `--threshold` (défaut 0.75)

3. **Analyse de sentiment** (`audio_analyzer/analyzer.py`)
   - Le texte agrégé de chaque locuteur est envoyé à Ollama
   - Retourne un JSON `{label: {sentiment, score, explication}}`
   - Fallback automatique si le modèle ne respecte pas le format JSON

4. **Résumé** (`audio_analyzer/analyzer.py`)
   - Transcript complet + sentiments → Ollama → résumé structuré en 5 sections

### Persistance (SQLite)

Schéma dans `database.py:init_db()` — 4 tables :
- `fingerprints` : empreintes nommées (embedding stocké en BLOB float32)
- `recordings` : métadonnées + transcript complet + résumé
- `speakers` : locuteurs par enregistrement, avec lien optionnel vers `fingerprints`
- `segments` : segments de parole individuels avec timestamps

### Configuration

Toute la config passe par `.env` (copier `.env.example`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | URL du serveur Ollama |
| `OLLAMA_MODEL` | `llama3` | Modèle utilisé pour sentiment + résumé |
| `WHISPER_MODEL_SIZE` | `base` | Taille du modèle Whisper (`tiny` → `large-v3`) |
| `AUDIO_LANGUAGE` | *(vide)* | Langue forcée, sinon auto-détection |
| `DB_PATH` | `audio_analysis.db` | Chemin SQLite |

### Points d'attention

- Le `VoiceEncoder` de resemblyzer est un singleton (`_encoder`) partagé entre `transcriber.py` et `fingerprint.py` — chaque module instancie le sien, ils ne se partagent pas l'état.
- `AgglomerativeClustering` avec `distance_threshold=0.35` détermine automatiquement le nombre de locuteurs. Passer `--speakers N` force `n_clusters=N` (plus fiable quand on connaît le nombre exact).
- Le fallback JSON dans `analyzer.py` retourne `"neutre"` / score `0.0` sans planter le pipeline si Ollama renvoie du texte non-JSON.
- `webrtcvad-wheels` est requis à la place de `webrtcvad` (pas de compilateur C++ nécessaire sur Windows).
