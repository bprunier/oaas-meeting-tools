#!/bin/bash
# Déploiement complet de OAAS Meeting Tools sur Linux

set -euo pipefail

# ── Couleurs ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()      { echo -e "${GREEN}[ OK ]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
die()     { echo -e "${RED}[ERR ]${NC}  $1"; exit 1; }

echo ""
echo "══════════════════════════════════════════════════════"
echo "   OAAS Meeting Tools — Déploiement Linux"
echo "══════════════════════════════════════════════════════"
echo ""

# ── 1. Python 3.10+ ───────────────────────────────────────────────────────────
info "Python..."
command -v python3 &>/dev/null || die "Python 3 introuvable. Installez Python 3.10+."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MIN=$(python3 -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)")
[ "$PY_MIN" -ge 310 ] || die "Python 3.10+ requis (trouvé : $PY_VER)"
ok "Python $PY_VER"

# ── 2. ffmpeg ─────────────────────────────────────────────────────────────────
info "ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg absent — installation..."
    if command -v apt-get &>/dev/null;  then sudo apt-get install -y ffmpeg
    elif command -v dnf &>/dev/null;    then sudo dnf install -y ffmpeg
    elif command -v pacman &>/dev/null; then sudo pacman -S --noconfirm ffmpeg
    else warn "Impossible d'installer ffmpeg automatiquement. Installez-le manuellement."; fi
fi
command -v ffmpeg &>/dev/null && ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | grep -oP '\d+\.\d+[\.\d]*')" \
    || warn "ffmpeg toujours absent — certains formats audio (mp3, m4a) peuvent échouer"

# ── 3. GPU ────────────────────────────────────────────────────────────────────
info "GPU NVIDIA..."
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU : $GPU"
else
    warn "Aucun GPU NVIDIA — analyse sur CPU (plus lent pour Whisper large-v3)"
fi

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
info "Ollama..."
if ! command -v ollama &>/dev/null; then
    info "Installation d'Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installé"
else
    ok "Ollama $(ollama --version 2>/dev/null | head -1)"
fi

# Démarrer Ollama si pas déjà en route
if ! ollama list &>/dev/null 2>&1; then
    info "Démarrage d'Ollama en arrière-plan..."
    ollama serve &>/dev/null &
    sleep 4
fi

# ── 5. Modèles Ollama ─────────────────────────────────────────────────────────
info "Modèle LLM : llama3..."
if ollama list 2>/dev/null | grep -q "^llama3"; then
    ok "llama3 déjà présent"
else
    ollama pull llama3 && ok "llama3 téléchargé"
fi

info "Modèle embeddings : nomic-embed-text..."
if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    ok "nomic-embed-text déjà présent"
else
    ollama pull nomic-embed-text && ok "nomic-embed-text téléchargé"
fi

# ── 6. Environnement virtuel Python ───────────────────────────────────────────
info "Environnement virtuel..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "venv créé"
else
    ok "venv existant"
fi
source .venv/bin/activate

# ── 7. Dépendances Python ─────────────────────────────────────────────────────
info "Dépendances Python..."
pip install --upgrade pip --quiet

# resemblyzer s'installe sans ses dépendances pour éviter les conflits de compilation
pip install --no-deps "resemblyzer>=0.1.1.dev0" --quiet

# Reste des dépendances
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
    "scikit-learn>=1.3.0" \
    "webrtcvad-wheels>=2.0.10" \
    "nvidia-cublas-cu12>=12.0.0" \
    "chromadb>=0.5.0" \
    --quiet

ok "Dépendances installées"

# ── 8. Configuration ──────────────────────────────────────────────────────────
info "Configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    ok ".env créé depuis .env.example"
else
    ok ".env existant (non écrasé)"
fi

# ── 9. Résumé ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "   Déploiement terminé"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Activer l'environnement :"
echo "    source .venv/bin/activate"
echo ""
echo "  Créer un projet et analyser :"
echo "    python main.py project create <nom>"
echo "    python main.py project use <nom>"
echo "    python main.py analyze <fichier.wav>"
echo ""
echo "  Ollama doit tourner avant toute analyse :"
echo "    ollama serve"
echo ""
