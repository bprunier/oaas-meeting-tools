# OAAS Meeting Tools

Pipeline d'analyse de réunions audio — transcription, diarisation, sentiment, résumé et recherche sémantique, 100% local, aucun compte requis.

## Fonctionnalités

- Transcription avec timestamps mot par mot (Whisper large-v3)
- Diarisation des locuteurs (qui parle quand)
- Identification par empreinte vocale
- Analyse de sentiment par locuteur
- Résumé automatique de réunion
- Détection de patterns : disponibilité, hébergement, qualité audio
- Recherche par mots-clés avec validation Ollama
- Recherche sémantique en langage naturel (RAG, 100% local via Ollama)
- Export ICS (Google Calendar) et CSV
- Gestion par projets — données isolées par projet

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) (LLM local)
- Linux avec support audio (CUDA recommandé)

## Installation

```bash
git clone <repository-url>
cd oaas-meeting-tools
```

**Linux :**
```bash
chmod +x deploy.sh && ./deploy.sh
```

**Windows** (PowerShell en tant qu'administrateur) :
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\deploy.ps1
```

Les scripts installent automatiquement ffmpeg, Ollama, les modèles (`llama3`, `nomic-embed-text`), le venv Python et créent le `.env`.

## Démarrage rapide

```bash
source .venv/bin/activate

# Créer et activer un projet
python main.py project create monProjet
python main.py project use monProjet

# Analyser un fichier audio
python main.py analyze reunion.wav

# Poser une question sur les réunions
python main.py ask "de quoi avez-vous parlé ?"
```

## Commandes

### Projets

Chaque projet est isolé dans `projects/<nom>/` avec sa propre BDD, index RAG, empreintes et exports.

```bash
python main.py project create <nom>       # créer un projet
python main.py project list               # lister (● = actif)
python main.py project use <nom>          # activer
python main.py project unset              # désactiver

# Forcer un projet pour une commande sans changer l'actif
python main.py --project <nom> <commande>
```

```
projects/<nom>/
├── audio_analysis.db   # base SQLite
├── chroma_db/          # index vectoriel RAG
├── voices/             # fichiers audio pour les empreintes
├── exports/            # CSV et ICS générés ici par défaut
└── .env               # surcharges locales (Whisper, Ollama...)
```

### Analyse audio

```bash
# Analyser un fichier (formats : mp3, wav, m4a, ogg, flac, opus, aac)
python main.py analyze <fichier> [--speakers N] [--threshold 0.75] [--vad-top-db 30]

# Avec détection de profils pour un locuteur identifié (busy, sleeping, qualité)
python main.py analyze <fichier> --profile-fingerprint <FP_ID>

# Scanner un répertoire (fichiers déjà analysés ignorés automatiquement)
python main.py scan-dir <répertoire> [--speakers N] [--threshold 0.75]
python main.py scan-dir <répertoire> --recursive
python main.py scan-dir <répertoire> --profile-fingerprint <FP_ID>
```

### Empreintes vocales

```bash
python main.py add-fingerprint "Nom" <fichier.wav>    # enregistrer une empreinte
python main.py fingerprints                            # lister
python main.py remove-fingerprint <id>                 # supprimer

# Améliorer la détection d'un locuteur à la voix variable :
python main.py extract-speaker-audio <id> [--output speaker.wav] [--min-duration 0.5]
python main.py enrich-fingerprint <id> <fichier.wav>
python main.py set-fingerprint-threshold <id> 0.60    # seuil personnalisé
python main.py set-fingerprint-threshold <id>         # reset au seuil global
```

### Consultation

```bash
python main.py list
python main.py show <id> [--no-transcript]
```

### Recherche par mots-clés

```bash
python main.py search "texte à chercher"
python main.py search "texte" --no-confirm            # sans validation Ollama (plus rapide)
python main.py search "ça coup(e|ait)" --regex        # expression régulière Python
python main.py search "texte" --speaker "Alice"       # filtrer par locuteur
python main.py search "texte" --recording <id>        # limiter à un enregistrement

# Profils prédéfinis
python main.py search --profile busy                  # détecte les indisponibilités
python main.py search --profile sleeping              # détecte les hébergements
python main.py search --profile quality               # détecte les problèmes audio
python main.py search --profile busy --speaker "Alice"
python main.py search --profile quality --recording <id>
```

| Profil | Détecte |
|--------|---------|
| `busy` | "occupé", "pas disponible", "j'ai pas le temps"… |
| `sleeping` | "dormir chez", "passer la nuit", "je reste chez"… |
| `quality` | "ça coupe", "tu m'entends", "j'entends pas"… |

### Recherche sémantique (RAG)

```bash
# Indexer les enregistrements (une fois après install, puis après import en masse)
python main.py index

# Poser une question en langage naturel
python main.py ask "de quoi avez-vous parlé ?"
python main.py ask "y a-t-il eu des problèmes techniques ?" --top-k 15
python main.py ask "quel est le sujet principal ?" --recording <id>
```

Les nouveaux enregistrements sont indexés automatiquement à la fin de chaque `analyze`.

### Export & maintenance

```bash
python main.py export-csv [--output rapport.csv]
python main.py export-ics [<id> ...] [--output calendrier.ics]

python main.py backfill-dates                                    # remplir les dates manquantes
python main.py backfill-detections --profile quality             # (ré)analyser tous les enreg.
python main.py backfill-detections --profile busy --fingerprint <FP_ID>

python main.py clear-recordings [--yes]                          # supprimer tous les enreg. (empreintes conservées)
```

## Configuration

`.env` global — s'applique à tous les projets :

| Variable | Défaut | Rôle |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3` | Taille du modèle Whisper (`tiny` → `large-v3`) |
| `AUDIO_LANGUAGE` | `fr` | Langue forcée (vide = auto-détection) |
| `MIN_SPEAKING_TIME` | `5` | Secondes min. pour figurer dans la synthèse |
| `VAD_TOP_DB` | `35` | Seuil VAD — baisser pour capter les voix faibles |
| `OLLAMA_HOST` | `http://localhost:11434` | URL du serveur Ollama |
| `OLLAMA_MODEL` | `llama3` | Modèle LLM (sentiment, résumé, RAG) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modèle embeddings RAG via Ollama |

`DB_PATH` et `CHROMA_PATH` sont gérés automatiquement par le système de projets.

Chaque projet peut surcharger ces variables dans `projects/<nom>/.env`.

## Structure du projet

```
oaas-meeting-tools/
├── main.py                    # Point d'entrée
├── requirements.txt
├── deploy.sh                  # Script de déploiement Linux
├── deploy.ps1                 # Script de déploiement Windows
├── .env.example               # Template de configuration
├── audio_analyzer/
│   ├── config.py              # Variables de configuration
│   ├── database.py            # Opérations SQLite
│   ├── transcriber.py         # Transcription + diarisation
│   ├── analyzer.py            # Sentiment + résumé (Ollama)
│   ├── fingerprint.py         # Empreintes vocales
│   ├── voice_encoder.py       # Singleton resemblyzer (forcé CPU)
│   ├── embedder.py            # Embeddings Ollama + ChromaDB (RAG)
│   ├── project.py             # Gestion des projets
│   ├── searcher.py            # Recherche keyword + profils
│   ├── date_detector.py       # Détection de date
│   └── ics_exporter.py        # Export ICS
└── projects/                  # Données par projet (gitignore)
    └── <nom>/
        ├── audio_analysis.db
        ├── chroma_db/
        ├── voices/
        ├── exports/
        └── .env
```

## Licence

MIT
