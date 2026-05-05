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
from audio_analyzer.date_detector import detect_recording_date
from audio_analyzer.ics_exporter import build_ics


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
    recording_date = detect_recording_date(audio_path, full_transcript)
    rec_id = db.save_recording(audio_path, duration, full_transcript, recording_date)
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
    print(f"Durée        : {fmt_time(duration)}")
    print(f"Date         : {recording_date or 'non détectée'}\n")

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
        date_str = r['recording_date'] or '?'
        print(f"  [{r['id']:>4}] {Path(r['filename']).name:<40} "
              f"{fmt_time(r['duration'] or 0):>8}  "
              f"{r['speaker_count']} locuteur(s)  "
              f"enreg.: {date_str}")
    print()


# ── Commande : show ───────────────────────────────────────────────────────────

def cmd_show(args):
    data = db.get_recording(args.id)
    if not data:
        print(f"Enregistrement {args.id} introuvable.")
        sys.exit(1)

    rec = data["recording"]
    print_header(f"Enregistrement #{rec['id']} – {Path(rec['filename']).name}")
    date_str = rec['recording_date'] or 'non détectée'
    print(f"Durée : {fmt_time(rec['duration'] or 0)}  |  Date : {date_str}  |  {rec['created_at'][:16]}\n")

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


# ── Commande : backfill-dates ─────────────────────────────────────────────────

def cmd_backfill_dates(args):
    rows = db.get_recordings_without_date()
    if not rows:
        print("Tous les enregistrements ont déjà une date.")
        return

    updated, skipped = 0, 0
    for row in rows:
        d = detect_recording_date(row['filename'], row['transcript_full'] or '')
        if d:
            db.update_recording_date(row['id'], d)
            print(f"  [{row['id']:>4}] {d}  ← {Path(row['filename']).name}")
            updated += 1
        else:
            print(f"  [{row['id']:>4}] non détectée  ← {Path(row['filename']).name}")
            skipped += 1

    print(f"\n{updated} mise(s) à jour, {skipped} non résolue(s).")


# ── Commande : export-ics ────────────────────────────────────────────────────

def cmd_export_ics(args):
    if args.ids:
        recordings_data = []
        for rid in args.ids:
            data = db.get_recording(rid)
            if not data:
                print(f"Enregistrement {rid} introuvable, ignoré.")
                continue
            if not (data['recording'].get('recording_date') or data['recording'].get('summary')):
                print(f"Enregistrement {rid} sans date ni résumé, ignoré.")
                continue
            recordings_data.append(data)
    else:
        all_recs = db.list_recordings()
        recordings_data = [db.get_recording(r['id']) for r in all_recs]
        recordings_data = [d for d in recordings_data if d]

    if not recordings_data:
        print("Aucun enregistrement à exporter.")
        return

    ics_content = build_ics(recordings_data)
    output = args.output or 'recordings.ics'
    Path(output).write_text(ics_content, encoding='utf-8')
    print(f"{len(recordings_data)} enregistrement(s) exporté(s) → {output}")


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

    # backfill-dates
    sub.add_parser("backfill-dates", help="Remplir les dates manquantes des enregistrements existants")

    # export-ics
    p_ics = sub.add_parser("export-ics", help="Exporter les réunions au format ICS (Google Calendar)")
    p_ics.add_argument(
        "ids", nargs="*", type=int,
        help="IDs des enregistrements à exporter (tous si omis)"
    )
    p_ics.add_argument(
        "--output", "-o", default="recordings.ics",
        help="Fichier de sortie (défaut: recordings.ics)"
    )

    args = parser.parse_args()

    commands = {
        "analyze": cmd_analyze,
        "add-fingerprint": cmd_add_fingerprint,
        "list": cmd_list,
        "show": cmd_show,
        "fingerprints": cmd_fingerprints,
        "export-ics": cmd_export_ics,
        "backfill-dates": cmd_backfill_dates,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
