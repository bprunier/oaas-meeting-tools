"""
Transcription (faster-whisper) + diarisation locale sans compte externe.
Pipeline : librosa VAD → resemblyzer embeddings → clustering agglomératif → Whisper
"""
from __future__ import annotations
import numpy as np
import torch
import librosa
from faster_whisper import WhisperModel
from resemblyzer import preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from audio_analyzer.config import WHISPER_MODEL_SIZE, AUDIO_LANGUAGE
from audio_analyzer.voice_encoder import get_encoder

SAMPLE_RATE = 16000


def _load_whisper(force_cpu: bool = False) -> tuple[WhisperModel, str]:
    if force_cpu:
        return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"), "CPU"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        label = "GPU" if device == "cuda" else "CPU"
        return WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute), label
    except RuntimeError as e:
        if "libcublas" in str(e) or "CUDA" in str(e):
            print("  CUDA error at model load, falling back to CPU...")
            return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"), "CPU"
        raise


def _collect_segments(raw_segments, speaker_map: dict) -> list[dict]:
    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        speaker = _dominant_speaker(seg.start, seg.end, speaker_map)
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": speaker,
            "text": text,
        })
    return segments


def _vad_segments(waveform: np.ndarray, sr: int,
                  top_db: int = 35,
                  min_duration: float = 0.4) -> list[tuple[float, float]]:
    """Détecte les segments de parole via l'énergie (librosa). Aucun téléchargement."""
    intervals = librosa.effects.split(waveform, top_db=top_db,
                                      frame_length=1024, hop_length=256)
    return [
        (int(s) / sr, int(e) / sr)
        for s, e in intervals
        if (int(e) - int(s)) / sr >= min_duration
    ]


def _diarize(waveform: np.ndarray, sr: int,
             vad_segs: list[tuple[float, float]],
             num_speakers: int | None = None) -> dict[tuple, str]:
    """
    Assigne un label locuteur à chaque segment VAD via resemblyzer + clustering.
    Retourne {(start, end): "SPEAKER_XX"}
    """
    encoder = get_encoder()
    embeddings: list[np.ndarray] = []
    valid_segs: list[tuple[float, float]] = []

    for start, end in vad_segs:
        clip = waveform[int(start * sr): int(end * sr)]
        if len(clip) < int(0.4 * sr):
            continue
        try:
            processed = preprocess_wav(clip, source_sr=sr)
            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)
            valid_segs.append((start, end))
        except Exception:
            pass

    if not embeddings:
        return {}
    if len(embeddings) == 1:
        return {valid_segs[0]: "SPEAKER_00"}

    X = np.array(embeddings)

    if num_speakers is not None:
        n = min(num_speakers, len(embeddings))
        model = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.35,
            metric="cosine", linkage="average"
        )

    labels = model.fit_predict(X)
    return {seg: f"SPEAKER_{lbl:02d}" for seg, lbl in zip(valid_segs, labels)}


def _dominant_speaker(seg_start: float, seg_end: float,
                      speaker_map: dict[tuple, str]) -> str:
    """Trouve le locuteur dominant sur la plage temporelle d'un segment Whisper."""
    overlaps: dict[str, float] = {}
    for (s, e), label in speaker_map.items():
        overlap = min(seg_end, e) - max(seg_start, s)
        if overlap > 0:
            overlaps[label] = overlaps.get(label, 0) + overlap
    return max(overlaps, key=overlaps.get) if overlaps else "SPEAKER_00"


def transcribe(audio_path: str,
               num_speakers: int | None = None,
               vad_top_db: int = 35) -> tuple[list[dict], float]:
    """
    Retourne (segments, durée_secondes).
    Chaque segment : {start, end, speaker, text}

    num_speakers : nombre de locuteurs si connu, sinon détection automatique.
    """
    device_label = "GPU" if torch.cuda.is_available() else "CPU"

    print("  Chargement de l'audio...")
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(waveform) / SAMPLE_RATE

    print(f"  Détection des segments de parole (VAD) [CPU, top_db={vad_top_db}]...")
    vad_segs = _vad_segments(waveform, SAMPLE_RATE, top_db=vad_top_db)
    print(f"  {len(vad_segs)} segments détectés.")

    print(f"  Identification des locuteurs (clustering) [{device_label}]...")
    speaker_map = _diarize(waveform, SAMPLE_RATE, vad_segs, num_speakers)
    n_speakers = len(set(speaker_map.values()))
    print(f"  {n_speakers} locuteur(s) identifié(s).")

    whisper, whisper_device = _load_whisper()
    print(f"  Transcription Whisper [{whisper_device}]...")
    kwargs: dict = {
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "temperature": 0.0,
    }
    if AUDIO_LANGUAGE:
        kwargs["language"] = AUDIO_LANGUAGE

    raw_segments, info = whisper.transcribe(audio_path, **kwargs)
    try:
        segments = _collect_segments(raw_segments, speaker_map)
    except RuntimeError as e:
        if "libcublas" not in str(e) and "CUDA" not in str(e):
            raise
        print("  CUDA error during transcription, retrying on CPU...")
        whisper, _ = _load_whisper(force_cpu=True)
        raw_segments, info = whisper.transcribe(audio_path, **kwargs)
        segments = _collect_segments(raw_segments, speaker_map)

    return segments, round(info.duration, 2)
