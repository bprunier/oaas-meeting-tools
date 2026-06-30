# CLAUDE.md

Ce fichier documente le projet pour Claude Code.

## Prérequis

```bash
# Ollama doit tourner avant toute analyse ou question RAG
ollama serve
ollama pull llama3           # LLM principal
ollama pull nomic-embed-text # Embeddings RAG
```

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Gestion des projets

Chaque projet est isolé dans `projects/<nom>/` (gitignore).

```bash
python main.py project create <nom>   # créer un projet (BDD, chroma, voices, exports, .env)
python main.py project list           # lister (● = actif)
python main.py project use <nom>      # activer (persisté dans .current_project)
python main.py project unset          # désactiver

# Toutes les commandes utilisent le projet actif automatiquement.
# Pour forcer un projet sans changer l'actif :
python main.py --project <nom> <commande>
```

Structure d'un projet :
```
projects/<nom>/
├── audio_analysis.db   # BDD SQLite
├── chroma_db/          # index vectoriel RAG
├── voices/             # fichiers audio pour add-fingerprint
├── exports/            # CSV et ICS (destination par défaut)
└── .env               # surcharges locales (Whisper, Ollama, etc.)
```

## Analyse audio

```bash
python main.py analyze <fichier.mp3> [--speakers N] [--threshold 0.75] [--vad-top-db 30]
python main.py analyze <fichier.mp3> --profile-fingerprint <FP_ID>

# Scanner un répertoire (saute les fichiers déjà analysés)
python main.py scan-dir <répertoire> [--speakers N] [--recursive]
```

## Empreintes vocales

```bash
python main.py add-fingerprint "Nom" <fichier.wav>
python main.py fingerprints
python main.py remove-fingerprint <id>
python main.py enrich-fingerprint <id> <fichier.wav>          # améliorer une empreinte
python main.py extract-speaker-audio <id> [--output out.wav]  # extraire les segments d'un locuteur
python main.py set-fingerprint-threshold <id> 0.60            # seuil par locuteur
python main.py set-fingerprint-threshold <id>                 # reset au seuil global
```

## Consultation des résultats

```bash
python main.py list
python main.py show <id> [--no-transcript]
```

## Recherche

```bash
# Recherche texte (keyword)
python main.py search "occupé"
python main.py search "occupé" --no-confirm          # sans validation Ollama
python main.py search "ça coup(e|ait)" --regex
python main.py search --speaker "Alice"
python main.py search --recording <id>

# Profils prédéfinis : busy | sleeping | quality
python main.py search --profile busy
python main.py search --profile quality --recording <id>
```

## Recherche sémantique (RAG)

```bash
# Indexer (une fois après install, ou après import en masse)
python main.py index

# Poser une question en langage naturel
python main.py ask "de quoi avez-vous parlé ?"
python main.py ask "y a-t-il eu des problèmes techniques ?" --top-k 15
python main.py ask "quel est le sujet ?" --recording <id>
```

Les nouveaux enregistrements sont indexés automatiquement après chaque `analyze`.

## Export & maintenance

```bash
python main.py export-csv [--output fichier.csv]
python main.py export-ics [<id> ...] [--output fichier.ics]
python main.py backfill-dates
python main.py backfill-detections --profile quality
python main.py backfill-detections --profile busy --fingerprint <FP_ID>
python main.py clear-recordings [--yes]
```

## Configuration

`.env` global (config infrastructure, tous projets) :

| Variable | Défaut | Rôle |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3` | Taille du modèle Whisper |
| `AUDIO_LANGUAGE` | `fr` | Langue forcée (vide = auto) |
| `MIN_SPEAKING_TIME` | `5` | Secondes min. pour figurer dans la synthèse |
| `VAD_TOP_DB` | `35` | Seuil VAD (baisser pour voix faibles) |
| `OLLAMA_HOST` | `http://localhost:11434` | URL Ollama |
| `OLLAMA_MODEL` | `llama3` | Modèle LLM |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modèle embeddings RAG (via Ollama) |
| `RAG_TOP_K` | `20` | Nombre de segments récupérés par défaut pour `ask` |

`DB_PATH` et `CHROMA_PATH` sont gérés automatiquement par le système de projets — ne pas les définir dans `.env`.

`projects/<nom>/.env` peut surcharger n'importe quelle variable ci-dessus pour ce projet uniquement.

## Architecture

Pipeline orchestré dans `main.py:_analyze_file` :

1. **Transcription + diarisation** (`transcriber.py`)
   - VAD énergie librosa → resemblyzer (CPU) → AgglomerativeClustering cosine → Whisper large-v3 (GPU)
   - `_dominant_speaker` fusionne timestamps Whisper × clusters locuteurs

2. **Identification des locuteurs** (`fingerprint.py`)
   - Similarité cosinus contre les empreintes nommées en base (seuil global ou par locuteur)

3. **Sentiment** (`analyzer.py`)
   - Texte agrégé par locuteur → Ollama → JSON `{label: {sentiment, score, explication}}`

4. **Résumé** (`analyzer.py`)
   - Transcript + sentiments + participants → Ollama → résumé 6 sections

5. **Détection de date** (`date_detector.py`)
   - Nom de fichier → transcript → métadonnées audio

6. **Export ICS** (`ics_exporter.py`) — RFC 5545, importable Google Calendar

7. **Indexation RAG** (`embedder.py`)
   - Ollama `EMBEDDING_MODEL` (GPU via Ollama) → ChromaDB `PersistentClient` dans `projects/<nom>/chroma_db/`
   - Changer de modèle d'embedding nécessite de ré-indexer (supprimer `chroma_db/` puis `python main.py index`)

8. **Projets** (`project.py`)
   - `activate_project(nom)` met à jour `config.DB_PATH` et `config.CHROMA_PATH` avant `init_db()`
   - `database.py` et `embedder.py` lisent les chemins dynamiquement via `config`

### Persistance (SQLite — par projet)

4 tables dans `database.py:init_db()` :
- `fingerprints` — empreintes nommées (embedding BLOB float32)
- `recordings` — métadonnées + transcript + résumé + `recording_date`
- `speakers` — locuteurs par enregistrement, lien optionnel vers `fingerprints`
- `segments` — segments de parole avec timestamps

### Points d'attention

- `VoiceEncoder` resemblyzer forcé sur **CPU** (`voice_encoder.py`) — évite OOM CUDA avec Whisper large-v3
- `AgglomerativeClustering` avec `distance_threshold=0.35` détermine le nombre de locuteurs automatiquement ; `--speakers N` force `n_clusters=N`
- Fallback JSON dans `analyzer.py` → `"neutre"` / score `0.0` si Ollama répond hors format
- `webrtcvad-wheels` requis à la place de `webrtcvad` (pas de compilateur C++ sur Windows)
