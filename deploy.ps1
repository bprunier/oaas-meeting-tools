# Déploiement complet de OAAS Meeting Tools sur Windows
# Lancer depuis PowerShell en tant qu'administrateur :
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\deploy.ps1

$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Info($msg)    { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Ok($msg)      { Write-Host "[ OK ]  $msg" -ForegroundColor Green }
function Warn($msg)    { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Fail($msg)    { Write-Host "[ERR ]  $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "   OAAS Meeting Tools — Déploiement Windows"          -ForegroundColor White
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""

# ── 1. Python 3.10+ ───────────────────────────────────────────────────────────
Info "Python..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "Python introuvable. Installez Python 3.10+ depuis https://www.python.org/downloads/"
}
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pyMin = python -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)"
if ([int]$pyMin -lt 310) {
    Fail "Python 3.10+ requis (trouvé : $pyVer)"
}
Ok "Python $pyVer"

# ── 2. ffmpeg ─────────────────────────────────────────────────────────────────
Info "ffmpeg..."
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Warn "ffmpeg absent — tentative d'installation via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Gyan.FFmpeg --silent --accept-source-agreements --accept-package-agreements
        # Rafraîchir le PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Warn "winget indisponible. Installez ffmpeg manuellement : https://ffmpeg.org/download.html"
        Warn "Certains formats audio (mp3, m4a) peuvent ne pas fonctionner sans ffmpeg."
    }
}
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $ffVer = (ffmpeg -version 2>&1 | Select-String -Pattern "ffmpeg version (\S+)").Matches.Groups[1].Value
    Ok "ffmpeg $ffVer"
} else {
    Warn "ffmpeg toujours absent après tentative d'installation"
}

# ── 3. GPU ────────────────────────────────────────────────────────────────────
Info "GPU NVIDIA..."
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
    Ok "GPU : $gpu"
} else {
    Warn "Aucun GPU NVIDIA détecté — analyse sur CPU (plus lent pour Whisper large-v3)"
}

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
Info "Ollama..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Info "Installation d'Ollama..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Fail "winget indisponible. Installez Ollama manuellement depuis https://ollama.com"
    }
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $olVer = (ollama --version 2>$null | Select-Object -First 1)
    Ok "Ollama $olVer"
} else {
    Fail "Ollama introuvable après installation. Relancez PowerShell et réessayez."
}

# Démarrer Ollama si pas en route
$ollamaRunning = $false
try { ollama list 2>$null | Out-Null; $ollamaRunning = $true } catch {}
if (-not $ollamaRunning) {
    Info "Démarrage d'Ollama en arrière-plan..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

# ── 5. Modèles Ollama ─────────────────────────────────────────────────────────
Info "Modèle LLM : llama3..."
$models = ollama list 2>$null
if ($models -match "llama3") {
    Ok "llama3 déjà présent"
} else {
    ollama pull llama3
    Ok "llama3 téléchargé"
}

Info "Modèle embeddings : nomic-embed-text..."
if ($models -match "nomic-embed-text") {
    Ok "nomic-embed-text déjà présent"
} else {
    ollama pull nomic-embed-text
    Ok "nomic-embed-text téléchargé"
}

# ── 6. Environnement virtuel Python ───────────────────────────────────────────
Info "Environnement virtuel..."
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Ok "venv créé"
} else {
    Ok "venv existant"
}

$pip = ".venv\Scripts\pip.exe"
$python = ".venv\Scripts\python.exe"

# ── 7. Dépendances Python ─────────────────────────────────────────────────────
Info "Dépendances Python..."
& $pip install --upgrade pip --quiet

# webrtcvad-wheels : version pré-compilée, évite de nécessiter Visual C++ Build Tools
& $pip install "webrtcvad-wheels>=2.0.10" --quiet

# resemblyzer en --no-deps pour éviter que pip tente de compiler webrtcvad depuis les sources
& $pip install --no-deps "resemblyzer>=0.1.1.dev0" --quiet

# Reste des dépendances
& $pip install `
    "faster-whisper>=1.0.0" `
    "ollama>=0.4.0" `
    "python-dotenv>=1.0.0" `
    "numpy>=1.24.0" `
    "torch>=2.0.0" `
    "torchaudio>=2.0.0" `
    "soundfile>=0.12.0" `
    "librosa>=0.10.0" `
    "scipy>=1.10.0" `
    "scikit-learn>=1.3.0" `
    "chromadb>=0.5.0" `
    --quiet

Ok "Dépendances installées"

# ── 8. Configuration ──────────────────────────────────────────────────────────
Info "Configuration..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Ok ".env créé depuis .env.example"
} else {
    Ok ".env existant (non écrasé)"
}

# ── 9. Résumé ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "   Déploiement terminé"                                -ForegroundColor Green
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""
Write-Host "  Activer l'environnement :"
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Créer un projet et analyser :"
Write-Host "    python main.py project create <nom>"
Write-Host "    python main.py project use <nom>"
Write-Host "    python main.py analyze <fichier.wav>"
Write-Host ""
Write-Host "  Ollama doit tourner avant toute analyse :"
Write-Host "    ollama serve"
Write-Host ""
