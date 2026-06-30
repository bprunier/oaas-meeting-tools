from resemblyzer import VoiceEncoder

_encoder: VoiceEncoder | None = None


def get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        # Forcé sur CPU : le LSTM de resemblyzer (~17 MB) entre en conflit mémoire
        # avec Whisper large-v3 sur le même GPU (fragmentation CUDA).
        _encoder = VoiceEncoder(device="cpu")
    return _encoder
