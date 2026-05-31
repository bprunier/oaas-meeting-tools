#!/usr/bin/env python3
"""
Audio Analyzer - Pipeline d'analyse de réunions audio (100% local, aucun compte requis)
Usage:
  python main.py analyze <fichier_audio> [--speakers N] [--threshold 0.75]
  python main.py add-fingerprint <nom> <fichier_audio>
  python main.py list
  python main.py show <recording_id>
  python main.py fingerprints
  python main.py remove-fingerprint <id>
"""
from __future__ import annotations
import argparse
import ctypes
import os
import site
import sys
from pathlib import Path
from collections import defaultdict

# ctranslate2 4.x was compiled for CUDA 12 and does dlopen("libcublas.so.12").
# On CUDA 13 systems, preload the cu12 lib by full path so the soname lands in
# the process library cache before faster_whisper/ctranslate2 are imported.
for _sp in site.getsitepackages():
    _lib = os.path.join(_sp, "nvidia", "cublas", "lib", "libcublas.so.12")
    if os.path.exists(_lib):
        ctypes.CDLL(_lib)
        break

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


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac"}


# ── Pipeline d'analyse (réutilisé par analyze et scan-dir) ───────────────────

def _analyze_file(audio_path: str, num_speakers: int | None, threshold: float) -> int:
    print_header(f"Analyse : {Path(audio_path).name}")

    print("\n[Étape 1/4] Transcription et diarisation...")
    segments, duration = transcribe(audio_path, num_speakers=num_speakers)
    print(f"  Durée : {fmt_time(duration)} | Segments : {len(segments)}")

    speaker_texts: dict[str, list[str]] = defaultdict(list)
    speaker_segs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for seg in segments:
        speaker_texts[seg["speaker"]].append(seg["text"])
        speaker_segs[seg["speaker"]].append((seg["start"], seg["end"]))

    speaking_times = {
        label: sum(e - s for s, e in segs)
        for label, segs in speaker_segs.items()
    }

    full_transcript = "\n".join(
        f"[{fmt_time(s['start'])}] {s['speaker']}: {s['text']}"
        for s in segments
    )

    print("\n[Étape 2/4] Identification des locuteurs...")
    speaker_ids = identify_speakers(audio_path, speaker_segs, threshold=threshold)

    print(f"\n[Étape 3/4] Analyse des sentiments [Ollama: {config.OLLAMA_MODEL}]...")
    speaker_text_joined = {label: " ".join(texts) for label, texts in speaker_texts.items()}
    sentiments = analyze_sentiments(speaker_text_joined)

    print(f"\n[Étape 4/4] Génération du résumé [Ollama: {config.OLLAMA_MODEL}]...")
    participants = {
        label: speaker_ids[label]["name"] if speaker_ids.get(label) else label
        for label in speaker_texts
        if speaking_times.get(label, 0) >= config.MIN_SPEAKING_TIME
    }
    summary = generate_summary(full_transcript, sentiments, participants)

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
        print(f"  {label} → {name}{conf}")
        print(f"    Temps de parole : {fmt_time(speaking_times.get(label, 0))}")
        print(f"    Sentiment       : {sent.get('sentiment', '?')} "
              f"(score: {sent.get('score', 0):+.2f})")
        print(f"    {sent.get('explication', '')}\n")

    print_divider()
    print("RÉSUMÉ\n")
    print(summary)
    print_divider()
    print(f"\nSauvegardé en base → ID {rec_id}  (python main.py show {rec_id})\n")

    return rec_id


# ── Commande : analyze ────────────────────────────────────────────────────────

def cmd_analyze(args):
    config.validate()
    if not Path(args.audio).exists():
        print(f"Erreur : fichier introuvable → {args.audio}")
        sys.exit(1)
    _analyze_file(args.audio, args.speakers, args.threshold)
    sys.exit(0)


# ── Commande : scan-dir ───────────────────────────────────────────────────────

def cmd_scan_dir(args):
    config.validate()
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Erreur : répertoire introuvable → {directory}")
        sys.exit(1)

    pattern = "**/*" if args.recursive else "*"
    audio_files = sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"Aucun fichier audio trouvé dans {directory}")
        return

    already_done = db.get_analyzed_filenames()
    to_process = [f for f in audio_files if str(f.resolve()) not in already_done]
    skipped = len(audio_files) - len(to_process)

    print(f"{len(audio_files)} fichier(s) trouvé(s) — "
          f"{skipped} déjà analysé(s), {len(to_process)} à traiter.")

    if not to_process:
        print("Rien à faire.")
        return

    ok, errors = 0, []
    for i, audio_path in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {audio_path.name}")
        try:
            _analyze_file(str(audio_path.resolve()), args.speakers, args.threshold)
            ok += 1
        except Exception as e:
            print(f"  ERREUR : {e}")
            errors.append((audio_path.name, str(e)))

    print_divider()
    print(f"Scan terminé : {ok} analysé(s), {len(errors)} erreur(s).")
    if errors:
        for name, err in errors:
            print(f"  - {name} : {err}")


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


# ── Commande : export-csv ─────────────────────────────────────────────────────

def cmd_export_csv(args):
    import csv
    rows = db.export_recordings_csv()
    if not rows:
        print("Aucun enregistrement à exporter.")
        return
    output = args.output or "recordings.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "date", "duration", "speakers", "transcription", "résumé"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} enregistrement(s) exporté(s) → {output}")


# ── Commande : clear-recordings ──────────────────────────────────────────────

def cmd_clear_recordings(args):
    if not args.yes:
        confirm = input("Supprimer TOUS les enregistrements (recordings, speakers, segments) ? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Annulé.")
            return
    count = db.clear_recordings()
    print(f"{count} enregistrement(s) supprimé(s). Les empreintes vocales sont conservées.")


# ── Commande : remove-fingerprint ────────────────────────────────────────────

def cmd_remove_fingerprint(args):
    fps = db.list_fingerprints()
    match = next((fp for fp in fps if fp['id'] == args.id), None)
    if not match:
        print(f"Empreinte id={args.id} introuvable.")
        sys.exit(1)
    db.delete_fingerprint(args.id)
    print(f"Empreinte '{match['name']}' (id={args.id}) supprimée.")


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

    # scan-dir
    p_scan = sub.add_parser("scan-dir", help="Analyser tous les fichiers audio d'un répertoire")
    p_scan.add_argument("directory", help="Répertoire à scanner")
    p_scan.add_argument("--speakers", type=int, default=None,
                        help="Nombre de locuteurs (sinon détection automatique)")
    p_scan.add_argument("--threshold", type=float, default=0.75,
                        help="Seuil de similarité pour l'identification (défaut: 0.75)")
    p_scan.add_argument("--recursive", "-r", action="store_true",
                        help="Scanner les sous-répertoires")

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

    # remove-fingerprint
    p_rm_fp = sub.add_parser("remove-fingerprint", help="Supprimer une empreinte vocale")
    p_rm_fp.add_argument("id", type=int, help="ID de l'empreinte (voir 'fingerprints')")

    # export-csv
    p_csv = sub.add_parser("export-csv", help="Exporter tous les enregistrements en CSV")
    p_csv.add_argument(
        "--output", "-o", default="recordings.csv",
        help="Fichier de sortie (défaut: recordings.csv)"
    )

    # clear-recordings
    p_clear = sub.add_parser("clear-recordings", help="Supprimer tous les enregistrements de la base")
    p_clear.add_argument(
        "--yes", "-y", action="store_true",
        help="Confirmer sans prompt interactif"
    )

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
        "scan-dir": cmd_scan_dir,
        "add-fingerprint": cmd_add_fingerprint,
        "list": cmd_list,
        "show": cmd_show,
        "fingerprints": cmd_fingerprints,
        "remove-fingerprint": cmd_remove_fingerprint,
        "export-csv": cmd_export_csv,
        "clear-recordings": cmd_clear_recordings,
        "export-ics": cmd_export_ics,
        "backfill-dates": cmd_backfill_dates,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
