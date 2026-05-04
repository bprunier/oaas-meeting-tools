# Installation Windows sans Visual C++ Build Tools
# resemblyzer depend de webrtcvad (source), on contourne avec webrtcvad-wheels

$pip = ".venv\Scripts\pip"

# 1. webrtcvad-wheels fournit le module webrtcvad sans compiler de C++
& $pip install webrtcvad-wheels

# 2. resemblyzer installe avec --no-deps pour eviter pip tente de builder webrtcvad
& $pip install --no-deps "resemblyzer>=0.1.1.dev0"

# 3. Tout le reste
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
    "scikit-learn>=1.3.0"
