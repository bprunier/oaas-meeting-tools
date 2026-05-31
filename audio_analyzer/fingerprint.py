"""
Empreinte vocale via resemblyzer.
Permet d'enregistrer des locuteurs connus et de les identifier dans un audio.
"""
from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa
from resemblyzer import preprocess_wav
from pathlib import Path
from audio_analyzer import database as db
from audio_analyzer.voice_encoder import get_encoder


def extract_embedding(audio_path: str) -> np.ndarray:
    """Extrait l'empreinte vocale (256-dim) d'un fichier audio."""
    wav = preprocess_wav(Path(audio_path))
    encoder = get_encoder()
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
    encoder = get_encoder()
    wav_processed = preprocess_wav(combined, source_sr=16000)
    return encoder.embed_utterance(wav_processed)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_speaker(embedding: np.ndarray,
                  fingerprints: list[dict],
                  threshold: float = 0.75) -> dict | None:
    """
    Compare une empreinte aux empreintes connues.
    Utilise le threshold par empreinte s'il est défini, sinon le threshold global.
    """
    best_match = None
    best_score = -1.0
    for fp in fingerprints:
        score = cosine_similarity(embedding, fp["embedding"])
        if score > best_score:
            best_score = score
            best_match = fp

    if best_match is None:
        return None
    effective_threshold = best_match.get("threshold") or threshold
    if best_score >= effective_threshold:
        return {"id": best_match["id"], "name": best_match["name"], "score": best_score}
    return None


_MAX_ENRICH_SEC = 300  # 5 minutes suffisent largement pour un bon embedding


def enrich_fingerprint(fp_id: int, audio_path: str) -> tuple[float, float, int]:
    """
    Moyenne le nouvel embedding avec l'existant (moyenne pondérée par sample_count).
    L'audio est tronqué à _MAX_ENRICH_SEC pour éviter les OOM sur GPU.
    Retourne (similarité_avant, similarité_après, nouveau_sample_count).
    """
    fps = db.load_fingerprints()
    existing = next((f for f in fps if f["id"] == fp_id), None)
    if not existing:
        raise ValueError(f"Empreinte {fp_id} introuvable.")

    wav = preprocess_wav(Path(audio_path))
    max_samples = int(_MAX_ENRICH_SEC * 16000)
    if len(wav) > max_samples:
        print(f"  Audio tronqué à {_MAX_ENRICH_SEC}s (sur {len(wav)/16000:.0f}s total).")
        wav = wav[:max_samples]
    encoder = get_encoder()
    new_emb = encoder.embed_utterance(wav)
    sim_before = cosine_similarity(existing["embedding"], new_emb)

    n = existing["sample_count"]
    merged = (existing["embedding"] * n + new_emb) / (n + 1)
    merged = merged / (np.linalg.norm(merged) + 1e-8)

    sim_after = cosine_similarity(merged, existing["embedding"])
    db.update_fingerprint_embedding(fp_id, merged, n + 1)
    return sim_before, sim_after, n + 1


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
    Pass 1 : identification par cluster individuel.
    Pass 2 : si >= 2 clusters restent inconnus, leurs segments sont poolés.
             Si le pool matche un fingerprint > 0.80, chaque cluster inconnu
             est re-testé individuellement avec un seuil abaissé à 0.60.
    """
    known = db.load_fingerprints()
    if not known:
        return {label: None for label in speaker_segments}

    results: dict[str, dict | None] = {}

    # ── Pass 1 : identification individuelle ──────────────────────────────────
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

    # ── Pass 2 : pool des inconnus ────────────────────────────────────────────
    unknown_labels = [lbl for lbl, m in results.items() if m is None]
    if len(unknown_labels) >= 2:
        pool_segs = [seg for lbl in unknown_labels for seg in speaker_segments[lbl]]
        pool_dur = sum(e - s for s, e in pool_segs)
        print(f"\n  [Pass 2] {len(unknown_labels)} cluster(s) inconnu(s) "
              f"({pool_dur:.1f}s au total) — test poolé...")
        try:
            pool_emb = extract_embedding_from_segments(audio_path, pool_segs)
            pool_match = match_speaker(pool_emb, known, threshold=0.80)
            if pool_match:
                print(f"  [Pass 2] Pool → {pool_match['name']} ({pool_match['score']:.2f}) "
                      f"— re-test individuel (seuil 0.60)...")
                for label in unknown_labels:
                    segs = speaker_segments[label]
                    dur = sum(e - s for s, e in segs)
                    try:
                        emb = extract_embedding_from_segments(audio_path, segs)
                        match = match_speaker(emb, known, threshold=0.60)
                        if match and match["id"] == pool_match["id"]:
                            results[label] = match
                            print(f"    {label} ({dur:.1f}s) → {match['name']} "
                                  f"({match['score']:.2f}) ✓")
                        else:
                            print(f"    {label} ({dur:.1f}s) → inconnu "
                                  f"(score insuffisant)")
                    except Exception:
                        print(f"    {label} ({dur:.1f}s) → inconnu (erreur embedding)")
            else:
                print(f"  [Pass 2] Pas de match suffisant (seuil 0.80).")
        except Exception as e:
            print(f"  [Pass 2] Erreur : {e}")

    return results
