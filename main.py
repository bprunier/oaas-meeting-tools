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
from audio_analyzer.fingerprint import register_fingerprint, identify_speakers, enrich_fingerprint
from audio_analyzer.date_detector import detect_recording_date
from audio_analyzer.ics_exporter import build_ics
from audio_analyzer.searcher import PROFILES, search as do_search, confirm_profile


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

def _analyze_file(audio_path: str, num_speakers: int | None, threshold: float,
                  profile_fingerprint: int | None = None,
                  vad_top_db: int = 35) -> int:
    print_header(f"Analyse : {Path(audio_path).name}")

    print("\n[Étape 1/4] Transcription et diarisation...")
    segments, duration = transcribe(audio_path, num_speakers=num_speakers,
                                    vad_top_db=vad_top_db)
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

    participants = {
        label: speaker_ids[label]["name"] if speaker_ids.get(label) else label
        for label in speaker_texts
        if speaking_times.get(label, 0) >= config.MIN_SPEAKING_TIME
    }

    recording_date = detect_recording_date(audio_path, full_transcript)
    rec_id = db.save_recording(audio_path, duration, full_transcript, recording_date)

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

    # ── Détections de profil ─────────────────────────────────────────────────
    detections: list[dict] = []
    if profile_fingerprint is not None:
        fname = Path(audio_path).name
        date_str = recording_date.isoformat() if recording_date else None

        fp_label = next(
            (lbl for lbl, m in speaker_ids.items() if m and m["id"] == profile_fingerprint),
            None
        )
        if fp_label:
            fp_name = speaker_ids[fp_label]["name"]
            fp_segs = [s for s in segments if s["speaker"] == fp_label]
            fp_sp_id = speaker_db_ids[fp_label]
            print(f"\n[Détections profil] Locuteur ciblé : {fp_name} ({fp_label})")

            for prof in ("busy", "sleeping"):
                result = confirm_profile(fp_segs, prof, fname, date_str)
                db.save_profile_detection(
                    rec_id, fp_sp_id, prof,
                    result.get("confirmed"), result.get("explanation"), result.get("key_passage")
                )
                icon = "✓" if result.get("confirmed") else "✗"
                print(f"  {PROFILES[prof]['label']:<30} {icon}  {result.get('explanation','')}")
                if result.get("confirmed"):
                    det = dict(result, profile=prof,
                               identified_name=fp_name, speaker_label=fp_label)
                    detections.append(det)
        else:
            print(f"\n[Détections profil] Empreinte {profile_fingerprint} non identifiée dans cet enregistrement.")

        # Qualité audio : toujours sur l'ensemble de l'enregistrement
        result = confirm_profile(segments, "quality", fname, date_str)
        db.save_profile_detection(rec_id, None, "quality",
                                  result.get("confirmed"), result.get("explanation"), result.get("key_passage"))
        icon = "✓" if result.get("confirmed") else "✗"
        print(f"  {PROFILES['quality']['label']:<30} {icon}  {result.get('explanation','')}")
        if result.get("confirmed"):
            detections.append(dict(result, profile="quality",
                                   identified_name=None, speaker_label="enregistrement"))

    print(f"\n[Étape 4/4] Génération du résumé [Ollama: {config.OLLAMA_MODEL}]...")
    summary = generate_summary(full_transcript, sentiments, participants,
                               detections if detections else None)
    db.update_recording_summary(rec_id, summary)

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
    _analyze_file(args.audio, args.speakers, args.threshold,
                  profile_fingerprint=args.profile_fingerprint,
                  vad_top_db=args.vad_top_db)
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
            _analyze_file(str(audio_path.resolve()), args.speakers, args.threshold,
                          profile_fingerprint=args.profile_fingerprint,
                          vad_top_db=args.vad_top_db)
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


# ── Commande : enrich-fingerprint ────────────────────────────────────────────

def cmd_enrich_fingerprint(args):
    fps = db.list_fingerprints()
    match = next((f for f in fps if f["id"] == args.id), None)
    if not match:
        print(f"Empreinte id={args.id} introuvable.")
        sys.exit(1)
    if not Path(args.audio).exists():
        print(f"Fichier introuvable → {args.audio}")
        sys.exit(1)
    print(f"Enrichissement de l'empreinte '{match['name']}' (id={args.id})...")
    sim_before, sim_after, count = enrich_fingerprint(args.id, args.audio)
    print(f"  Similarité avec l'ancien embedding : {sim_before:.3f}")
    print(f"  Embedding mis à jour ({count} échantillon(s) agrégé(s)).")


# ── Commande : set-fingerprint-threshold ─────────────────────────────────────

def cmd_set_fingerprint_threshold(args):
    fps = db.list_fingerprints()
    match = next((f for f in fps if f["id"] == args.id), None)
    if not match:
        print(f"Empreinte id={args.id} introuvable.")
        sys.exit(1)
    threshold = args.threshold
    if threshold is not None and not (0.0 < threshold < 1.0):
        print("Le seuil doit être entre 0.0 et 1.0 (ex: 0.60).")
        sys.exit(1)
    db.update_fingerprint_threshold(args.id, threshold)
    if threshold is None:
        print(f"Seuil de '{match['name']}' réinitialisé au seuil global.")
    else:
        print(f"Seuil de '{match['name']}' (id={args.id}) fixé à {threshold:.2f}.")


# ── Commande : extract-speaker-audio ─────────────────────────────────────────

def cmd_extract_speaker_audio(args):
    fps = db.list_fingerprints()
    match = next((f for f in fps if f["id"] == args.id), None)
    if not match:
        print(f"Empreinte id={args.id} introuvable.")
        sys.exit(1)

    import numpy as np
    import soundfile as sf
    import librosa

    segments = db.get_segments_for_fingerprint(args.id)
    if not segments:
        print(f"Aucun segment trouvé pour '{match['name']}' — analysez d'abord des enregistrements.")
        sys.exit(1)

    min_dur = args.min_duration
    print(f"Extraction des segments de '{match['name']}' ({len(segments)} segment(s))...")

    clips = []
    missing, short = 0, 0
    for seg in segments:
        dur = seg["end_time"] - seg["start_time"]
        if dur < min_dur:
            short += 1
            continue
        fpath = seg["filename"]
        if not Path(fpath).exists():
            missing += 1
            continue
        wav, _ = librosa.load(fpath, sr=16000, mono=True,
                              offset=seg["start_time"], duration=dur)
        clips.append(wav)

    if not clips:
        print(f"Aucun segment utilisable (min {min_dur}s). Essayez --min-duration 0.")
        sys.exit(1)

    combined = np.concatenate(clips)
    total_sec = len(combined) / 16000
    output = args.output or f"speaker_{args.id}_{match['name'].replace(' ', '_')}.wav"
    sf.write(output, combined, 16000)
    print(f"  {len(clips)} segment(s) extraits — durée totale : {fmt_time(total_sec)}")
    if short:
        print(f"  {short} segment(s) ignoré(s) (< {min_dur}s)")
    if missing:
        print(f"  {missing} fichier(s) audio introuvable(s)")
    print(f"  → {output}")
    print(f"\nPour enrichir l'empreinte : python main.py enrich-fingerprint {args.id} {output}")


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


# ── Commande : backfill-detections ───────────────────────────────────────────

def cmd_backfill_detections(args):
    config.validate()
    profile = args.profile
    fp_id = args.fingerprint

    # Résoudre le nom du locuteur ciblé si fingerprint donné
    fp_name = None
    if fp_id is not None:
        fps = db.list_fingerprints()
        match = next((f for f in fps if f["id"] == fp_id), None)
        if not match:
            print(f"Empreinte id={fp_id} introuvable (voir 'fingerprints').")
            sys.exit(1)
        fp_name = match["name"]

    recordings = db.list_recordings()
    if not recordings:
        print("Aucun enregistrement en base.")
        return

    # Ignorer les enregistrements déjà traités pour ce profil + ce locuteur
    to_process = []
    for r in recordings:
        if args.force:
            to_process.append(r)
            continue
        existing = db.get_profile_detections(r["id"])
        if fp_id is not None:
            already = any(
                d["profile"] == profile and d.get("identified_name") == fp_name
                for d in existing
            )
        else:
            already = any(d["profile"] == profile for d in existing)
        if not already:
            to_process.append(r)

    skipped = len(recordings) - len(to_process)
    who_str = f" pour {fp_name}" if fp_name else ""
    print_header(f"Détection '{PROFILES[profile]['label']}'{who_str} — {len(to_process)} enregistrement(s)")
    if skipped:
        print(f"  ({skipped} déjà traité(s), ignorés)\n")

    if not to_process:
        print("Rien à faire.")
        return

    done, confirmed_count, absent_count, errors = 0, 0, 0, []
    for i, rec in enumerate(to_process, 1):
        name = Path(rec["filename"]).name
        print(f"  [{i:>3}/{len(to_process)}] #{rec['id']} {name:<36}", end=" ", flush=True)
        try:
            if args.force:
                db.delete_profile_detections(rec["id"], profile, fp_name)

            data = db.get_recording(rec["id"])

            if fp_id is not None:
                # Cibler les segments du locuteur identifié par cette empreinte
                fp_sp_ids = {
                    sp["id"] for sp in data["speakers"]
                    if sp.get("fingerprint_id") == fp_id
                }
                if not fp_sp_ids:
                    print("— absent")
                    absent_count += 1
                    done += 1
                    continue
                segs = [s for s in data["segments"] if s["speaker_id"] in fp_sp_ids]
                sp_id = next(iter(fp_sp_ids))
            else:
                segs = data["segments"]
                sp_id = None

            result = confirm_profile(segs, profile, name, rec.get("recording_date"))
            db.save_profile_detection(
                rec["id"], sp_id, profile,
                result.get("confirmed"), result.get("explanation"), result.get("key_passage")
            )
            icon = "✓" if result.get("confirmed") else "✗"
            if result.get("confirmed"):
                confirmed_count += 1
            expl = (result.get("explanation") or "")[:52]
            print(f"{icon}  {expl}")
            done += 1
        except Exception as e:
            print(f"ERREUR : {e}")
            errors.append((rec["id"], str(e)))

    print_divider()
    msg = f"{done} traité(s) — {confirmed_count} confirmé(s)"
    if fp_id is not None:
        msg += f" — {absent_count} sans ce locuteur"
    msg += f" — {len(errors)} erreur(s)."
    print(msg)
    if errors:
        for rid, err in errors:
            print(f"  #{rid} : {err}")
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

    detections = db.get_profile_detections(args.id)
    if detections:
        print_divider()
        print("DÉTECTIONS AUTOMATIQUES\n")
        _PROFILE_LABELS = {"busy": "Occupé / Indisponible",
                           "sleeping": "Dort chez quelqu'un",
                           "quality": "Qualité audio"}
        for d in detections:
            icon = "✓" if d["confirmed"] == 1 else ("✗" if d["confirmed"] == 0 else "?")
            who = d.get("identified_name") or d.get("speaker_label") or "enregistrement"
            label = _PROFILE_LABELS.get(d["profile"], d["profile"])
            print(f"  {label:<26} {icon}  {who}")
            if d.get("explanation"):
                print(f"    {d['explanation']}")
            if d.get("key_passage"):
                print(f"    → « {d['key_passage']} »")
            print()

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
        threshold_str = f"  seuil={fp['threshold']:.2f}" if fp.get("threshold") else ""
        samples_str = f"  {fp.get('sample_count') or 1} échantillon(s)"
        print(f"  [{fp['id']:>3}] {fp['name']:<30} {fp['created_at'][:16]}{samples_str}{threshold_str}")
    print()


# ── Commande : export-csv ─────────────────────────────────────────────────────

_CSV_FIELDS = [
    "id", "date", "duration", "speakers",
    "occupé / activité", "dort chez quelqu'un", "qualité audio",
    "transcription", "résumé",
]
_CSV_SEP = "²"


def _csv_cell(value: object) -> str:
    text = str(value) if value is not None else ""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return text


def cmd_export_csv(args):
    rows = db.export_recordings_csv()
    if not rows:
        print("Aucun enregistrement à exporter.")
        return
    output = args.output or "recordings.csv"
    with open(output, "w", encoding="utf-8") as f:
        f.write(_CSV_SEP.join(_CSV_FIELDS) + "\n")
        for row in rows:
            f.write(_CSV_SEP.join(_csv_cell(row.get(field, "")) for field in _CSV_FIELDS) + "\n")
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


# ── Commande : search ─────────────────────────────────────────────────────────

def cmd_search(args):
    config.validate()

    profile = getattr(args, "profile", None)
    query = getattr(args, "query", None)
    speaker = getattr(args, "speaker", None)
    no_confirm = getattr(args, "no_confirm", False)

    if not profile and not query:
        print("Erreur : fournir un texte à chercher ou --profile.")
        print(f"Profils disponibles : {', '.join(PROFILES)}")
        sys.exit(1)

    label = PROFILES[profile]["label"] if profile else f'"{query}"'
    print_header(f"Recherche : {label}")
    if speaker:
        print(f"  Filtre locuteur : {speaker}\n")

    recording_id = getattr(args, "recording", None)

    try:
        results = do_search(
            query=query,
            profile=profile,
            speaker_filter=speaker,
            confirm=not no_confirm,
            recording_id=recording_id,
        )
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    if not results:
        print("Aucune correspondance trouvée.")
        return

    total_matches = 0
    confirmed_count = 0

    for entry in results:
        rec = entry["recording"]
        matches = entry["matches"]
        ollama = entry["ollama"]
        total_matches += len(matches)

        date_str = rec["recording_date"] or "?"
        print_divider()
        print(f"[#{rec['id']}] {Path(rec['filename']).name}  —  {date_str}\n")

        for seg in matches:
            name = seg.get("identified_name") or seg["speaker_label"]
            print(f"  [{fmt_time(seg['start_time'])}] {name}: {seg['text']}")

        if ollama is not None:
            print()
            confirmed = ollama.get("confirmed")
            if confirmed is True:
                confirmed_count += 1
                icon = "✓"
            elif confirmed is False:
                icon = "✗"
            else:
                icon = "?"
            print(f"  Ollama {icon} {ollama.get('explanation', '')}")
            kp = ollama.get("key_passage")
            if kp:
                print(f"  → « {kp} »")
        print()

    print_divider()
    msg = f"{total_matches} segment(s) dans {len(results)} enregistrement(s)"
    if not no_confirm:
        msg += f" — {confirmed_count} confirmé(s) par Ollama"
    print(msg)
    print()


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
    p_analyze.add_argument(
        "--profile-fingerprint", type=int, default=None, metavar="FP_ID",
        help="ID d'empreinte : détecte busy+sleeping pour ce locuteur, qualité pour tout l'enreg."
    )
    p_analyze.add_argument(
        "--vad-top-db", type=int, default=config.VAD_TOP_DB, metavar="DB",
        help=f"Seuil VAD en dB sous le pic (défaut .env: {config.VAD_TOP_DB}, plus bas = capte les sons faibles)"
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
    p_scan.add_argument(
        "--profile-fingerprint", type=int, default=None, metavar="FP_ID",
        help="ID d'empreinte : détecte busy+sleeping pour ce locuteur, qualité pour tout l'enreg."
    )
    p_scan.add_argument(
        "--vad-top-db", type=int, default=config.VAD_TOP_DB, metavar="DB",
        help=f"Seuil VAD en dB sous le pic (défaut .env: {config.VAD_TOP_DB}, plus bas = capte les sons faibles)"
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

    # enrich-fingerprint
    p_enrich = sub.add_parser("enrich-fingerprint",
                               help="Enrichir une empreinte avec un nouveau clip audio (moyenne pondérée)")
    p_enrich.add_argument("id", type=int, help="ID de l'empreinte à enrichir")
    p_enrich.add_argument("audio", help="Fichier audio supplémentaire")

    # set-fingerprint-threshold
    p_thr = sub.add_parser("set-fingerprint-threshold",
                            help="Définir un seuil de détection spécifique pour une empreinte")
    p_thr.add_argument("id", type=int, help="ID de l'empreinte")
    p_thr.add_argument("threshold", type=float, nargs="?", default=None,
                       help="Seuil (0.0–1.0). Omis = reset au seuil global.")

    # extract-speaker-audio
    p_ext = sub.add_parser("extract-speaker-audio",
                            help="Extraire et concaténer tous les segments audio d'un locuteur identifié")
    p_ext.add_argument("id", type=int, help="ID de l'empreinte")
    p_ext.add_argument("--output", "-o", default=None, help="Fichier WAV de sortie")
    p_ext.add_argument("--min-duration", type=float, default=0.5, metavar="SEC",
                       help="Durée minimale d'un segment à inclure (défaut: 0.5s)")

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

    # backfill-detections
    p_bfd = sub.add_parser(
        "backfill-detections",
        help="Lancer une détection de profil sur tous les enregistrements existants"
    )
    p_bfd.add_argument(
        "--profile", "-p", choices=list(PROFILES), default="quality",
        help="Profil à détecter (défaut: quality)"
    )
    p_bfd.add_argument(
        "--fingerprint", "-f", type=int, default=None, metavar="FP_ID",
        help="Limiter la détection aux segments d'un locuteur identifié par cette empreinte"
    )
    p_bfd.add_argument(
        "--force", action="store_true",
        help="Ré-analyser même les enregistrements déjà traités (écrase les résultats existants)"
    )

    # backfill-dates
    sub.add_parser("backfill-dates", help="Remplir les dates manquantes des enregistrements existants")

    # search
    p_search = sub.add_parser("search", help="Rechercher un pattern dans les transcripts")
    p_search.add_argument(
        "query", nargs="?", default=None,
        help="Texte libre à rechercher dans les segments"
    )
    p_search.add_argument(
        "--profile", "-p",
        choices=list(PROFILES),
        help=f"Profil prédéfini : {', '.join(PROFILES)}"
    )
    p_search.add_argument(
        "--speaker", "-s",
        default=None,
        help="Filtrer sur un locuteur (nom partiel)"
    )
    p_search.add_argument(
        "--no-confirm", action="store_true",
        help="Ne pas appeler Ollama pour confirmer les résultats"
    )
    p_search.add_argument(
        "--recording", "-r", type=int, default=None, metavar="ID",
        help="Limiter la recherche à un enregistrement précis (ID)"
    )

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
        "enrich-fingerprint": cmd_enrich_fingerprint,
        "set-fingerprint-threshold": cmd_set_fingerprint_threshold,
        "extract-speaker-audio": cmd_extract_speaker_audio,
        "export-csv": cmd_export_csv,
        "clear-recordings": cmd_clear_recordings,
        "export-ics": cmd_export_ics,
        "backfill-dates": cmd_backfill_dates,
        "backfill-detections": cmd_backfill_detections,
        "search": cmd_search,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
