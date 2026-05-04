#!/usr/bin/env python3
"""
Audio Analyzer - Pipeline d'analyse de réunions audio (100% local, aucun compte requis)
Usage:
  python main.py analyze <fichier_audio> [--speakers N] [--threshold 0.75]
  python main.py add-fingerprint <nom> <fichier_audio>
  python main.py list
  python main.py show <recording_id>
  python main.py fingerprints
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

from audio_analyzer import config, database as db
from audio_analyzer.transcriber import transcribe
from audio_analyzer.analyzer import analyze_sentiments, generate_summary
from audio_analyzer.fingerprint import register_fingerprint, identify_speakers


# ── Helpers d'affichage ──────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def print_divider(char="─", width=70):
    print(char * width)


def print_header(title: str):
    print_divider("═")
    print(f"  {title}")
    print_divider("═")


# ── Commande : analyze ────────────────────────────────────────────────────────

def cmd_analyze(args):
    config.validate()
    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"Erreur : fichier introuvable → {audio_path}")
        sys.exit(1)

    print_header(f"Analyse : {Path(audio_path).name}")

    # 1. Transcription + diarisation
    print("\n[Étape 1/4] Transcription et diarisation...")
    segments, duration = transcribe(audio_path, num_speakers=args.speakers)
    print(f"  Durée : {fmt_time(duration)} | Segments : {len(segments)}")

    # Grouper segments par locuteur
    speaker_texts: dict[str, list[str]] = defaultdict(list)
    speaker_segs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for seg in segments:
        speaker_texts[seg["speaker"]].append(seg["text"])
        speaker_segs[seg["speaker"]].append((seg["start"], seg["end"]))

    # Calcul du temps de parole
    speaking_times = {
        label: sum(e - s for s, e in segs)
        for label, segs in speaker_segs.items()
    }

    # Texte complet
    full_transcript = "\n".join(
        f"[{fmt_time(s['start'])}] {s['speaker']}: {s['text']}"
        for s in segments
    )

    # 2. Identification via empreintes
    print("\n[Étape 2/4] Identification des locuteurs...")
    speaker_ids = identify_speakers(audio_path, speaker_segs, threshold=args.threshold)

    # 3. Analyse de sentiment
    print("\n[Étape 3/4] Analyse des sentiments...")
    speaker_text_joined = {
        label: " ".join(texts) for label, texts in speaker_texts.items()
    }
    sentiments = analyze_sentiments(speaker_text_joined)

    # 4. Résumé
    print("\n[Étape 4/4] Génération du résumé...")
    summary = generate_summary(full_transcript, sentiments)

    # ── Persistance ──────────────────────────────────────────────────────────
    rec_id = db.save_recording(audio_path, duration, full_transcript)
    db.update_recording_summary(rec_id, summary)

    speaker_db_ids: dict[str, int] = {}
    for label in speaker_texts:
        match = speaker_ids.get(label)
        sent = sentiments.get(label, {})
        sp_id = db.save_speaker(
            recording_id=rec_id,
            label=label,
            name=match["name"] if match else None,
            fp_id=match["id"] if match else None,
            sentiment=sent.get("sentiment", ""),
            score=sent.get("score", 0.0),
            speaking_time=speaking_times.get(label, 0.0),
        )
        speaker_db_ids[label] = sp_id

    db.save_segments([
        {
            "recording_id": rec_id,
            "speaker_id": speaker_db_ids[s["speaker"]],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
        }
        for s in segments
    ])

    # ── Affichage résultats ──────────────────────────────────────────────────
    print_divider()
    print(f"\n{'RÉSULTAT':^70}")
    print(f"Recording ID : {rec_id}")
    print(f"Durée        : {fmt_time(duration)}\n")

    print_divider()
    print("LOCUTEURS IDENTIFIÉS\n")
    for label in sorted(speaker_texts.keys()):
        match = speaker_ids.get(label)
        sent = sentiments.get(label, {})
        name = match["name"] if match else "Inconnu"
        conf = f" ({match['score']:.0%})" if match else ""
        time_str = fmt_time(speaking_times.get(label, 0))
        print(f"  {label} → {name}{conf}")
        print(f"    Temps de parole : {time_str}")
        print(f"    Sentiment       : {sent.get('sentiment', '?')} "
              f"(score: {sent.get('score', 0):+.2f})")
        print(f"    {sent.get('explication', '')}\n")

    print_divider()
    print("RÉSUMÉ\n")
    print(summary)
    print_divider()
    print(f"\nSauvegardé en base → ID {rec_id}  (python main.py show {rec_id})\n")


# ── Commande : add-fingerprint ────────────────────────────────────────────────

def cmd_add_fingerprint(args):
    if not Path(args.audio).exists():
        print(f"Erreur : fichier introuvable → {args.audio}")
        sys.exit(1)
    fp_id = register_fingerprint(args.name, args.audio)
    print(f"Empreinte '{args.name}' enregistrée (id={fp_id}).")


# ── Commande : list ───────────────────────────────────────────────────────────

def cmd_list(args):
    recordings = db.list_recordings()
    if not recordings:
        print("Aucun enregistrement en base.")
        return
    print_header("Enregistrements analysés")
    for r in recordings:
        print(f"  [{r['id']:>4}] {Path(r['filename']).name:<40} "
              f"{fmt_time(r['duration'] or 0):>8}  "
              f"{r['speaker_count']} locuteur(s)  "
              f"{r['created_at'][:16]}")
    print()


# ── Commande : show ───────────────────────────────────────────────────────────

def cmd_show(args):
    data = db.get_recording(args.id)
    if not data:
        print(f"Enregistrement {args.id} introuvable.")
        sys.exit(1)

    rec = data["recording"]
    print_header(f"Enregistrement #{rec['id']} – {Path(rec['filename']).name}")
    print(f"Durée : {fmt_time(rec['duration'] or 0)}  |  {rec['created_at'][:16]}\n")

    print_divider()
    print("LOCUTEURS\n")
    for sp in data["speakers"]:
        name = sp["identified_name"] or "Inconnu"
        print(f"  {sp['speaker_label']} → {name}")
        print(f"    Sentiment : {sp['overall_sentiment']} ({sp['sentiment_score']:+.2f})")
        print(f"    Temps     : {fmt_time(sp['speaking_time'] or 0)}\n")

    if not args.no_transcript:
        print_divider()
        print("TRANSCRIPTION\n")
        for seg in data["segments"]:
            name = seg["identified_name"] or seg["speaker_label"]
            print(f"  [{fmt_time(seg['start_time'])}] {name}: {seg['text']}")
        print()

    print_divider()
    print("RÉSUMÉ\n")
    print(rec["summary"] or "(non disponible)")
    print()


# ── Commande : fingerprints ───────────────────────────────────────────────────

def cmd_fingerprints(args):
    fps = db.list_fingerprints()
    if not fps:
        print("Aucune empreinte enregistrée.")
        return
    print_header("Empreintes vocales enregistrées")
    for fp in fps:
        print(f"  [{fp['id']:>3}] {fp['name']:<30} enregistrée le {fp['created_at'][:16]}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db.init_db()

    parser = argparse.ArgumentParser(
        description="Audio Analyzer – Transcription, diarisation, sentiment, résumé"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyser un fichier audio")
    p_analyze.add_argument("audio", help="Chemin vers le fichier audio")
    p_analyze.add_argument(
        "--speakers", type=int, default=None,
        help="Nombre de locuteurs si connu (sinon détection automatique)"
    )
    p_analyze.add_argument(
        "--threshold", type=float, default=0.75,
        help="Seuil de similarité pour l'identification par empreinte (défaut: 0.75)"
    )

    # add-fingerprint
    p_fp = sub.add_parser("add-fingerprint", help="Enregistrer une empreinte vocale")
    p_fp.add_argument("name", help="Nom du locuteur")
    p_fp.add_argument("audio", help="Fichier audio contenant ce locuteur seul")

    # list
    sub.add_parser("list", help="Lister les enregistrements analysés")

    # show
    p_show = sub.add_parser("show", help="Afficher les détails d'un enregistrement")
    p_show.add_argument("id", type=int, help="ID de l'enregistrement")
    p_show.add_argument(
        "--no-transcript", action="store_true",
        help="Masquer la transcription complète"
    )

    # fingerprints
    sub.add_parser("fingerprints", help="Lister les empreintes vocales enregistrées")

    args = parser.parse_args()

    commands = {
        "analyze": cmd_analyze,
        "add-fingerprint": cmd_add_fingerprint,
        "list": cmd_list,
        "show": cmd_show,
        "fingerprints": cmd_fingerprints,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
