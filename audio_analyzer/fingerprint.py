"""
Empreinte vocale via resemblyzer.
Permet d'enregistrer des locuteurs connus et de les identifier dans un audio.
"""
from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from audio_analyzer import database as db

_encoder: VoiceEncoder | None = None


def _get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


def extract_embedding(audio_path: str) -> np.ndarray:
    """Extrait l'empreinte vocale (256-dim) d'un fichier audio."""
    wav = preprocess_wav(Path(audio_path))
    encoder = _get_encoder()
    return encoder.embed_utterance(wav)


def extract_embedding_from_segments(audio_path: str,
                                    segments: list[tuple[float, float]]) -> np.ndarray:
    """
    Extrait l'empreinte à partir de segments temporels précis (start, end en secondes).
    Utile pour n'extraire que les segments d'un locuteur spécifique.
    """
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    clips = []
    for start, end in segments:
        s = int(start * sr)
        e = int(end * sr)
        clips.append(wav[s:e])

    if not clips:
        raise ValueError("Aucun segment audio valide fourni.")

    combined = np.concatenate(clips)
    encoder = _get_encoder()
    wav_processed = preprocess_wav(combined, source_sr=16000)
    return encoder.embed_utterance(wav_processed)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_speaker(embedding: np.ndarray,
                  fingerprints: list[dict],
                  threshold: float = 0.75) -> dict | None:
    """
    Compare une empreinte aux empreintes connues.
    Retourne le meilleur match si > threshold, sinon None.
    """
    best_match = None
    best_score = -1.0
    for fp in fingerprints:
        score = cosine_similarity(embedding, fp["embedding"])
        if score > best_score:
            best_score = score
            best_match = fp

    if best_score >= threshold:
        return {"id": best_match["id"], "name": best_match["name"], "score": best_score}
    return None


def register_fingerprint(name: str, audio_path: str) -> int:
    """Enregistre l'empreinte d'un locuteur connu en base de données."""
    print(f"Extraction de l'empreinte pour '{name}'...")
    embedding = extract_embedding(audio_path)
    fp_id = db.save_fingerprint(name, embedding)
    print(f"Empreinte enregistrée (id={fp_id}, dim={len(embedding)}).")
    return fp_id


def identify_speakers(audio_path: str,
                       speaker_segments: dict[str, list[tuple[float, float]]],
                       threshold: float = 0.75) -> dict[str, dict | None]:
    """
    Pour chaque locuteur, extrait son empreinte depuis ses segments
    et tente de le matcher aux empreintes connues.

    speaker_segments : {speaker_label: [(start, end), ...]}
    Retourne : {speaker_label: match_info | None}
    """
    known = db.load_fingerprints()
    if not known:
        return {label: None for label in speaker_segments}

    results = {}
    for label, segs in speaker_segments.items():
        try:
            emb = extract_embedding_from_segments(audio_path, segs)
            match = match_speaker(emb, known, threshold)
            results[label] = match
            if match:
                print(f"  {label} → {match['name']} (similarité: {match['score']:.2f})")
            else:
                print(f"  {label} → inconnu (meilleur score < {threshold})")
        except Exception as e:
            print(f"  {label} → erreur empreinte: {e}")
            results[label] = None

    return results
