import sqlite3
import json
import numpy as np
from contextlib import contextmanager
from audio_analyzer.config import DB_PATH


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS recordings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT NOT NULL,
                duration        REAL,
                recording_date  DATE,
                transcript_full TEXT,
                summary         TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS speakers (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id        INTEGER REFERENCES recordings(id),
                speaker_label       TEXT NOT NULL,
                identified_name     TEXT,
                fingerprint_id      INTEGER REFERENCES fingerprints(id),
                overall_sentiment   TEXT,
                sentiment_score     REAL,
                speaking_time       REAL
            );

            CREATE TABLE IF NOT EXISTS segments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id    INTEGER REFERENCES recordings(id),
                speaker_id      INTEGER REFERENCES speakers(id),
                start_time      REAL,
                end_time        REAL,
                text            TEXT
            );
        """)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profile_detections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
                speaker_id   INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
                profile      TEXT NOT NULL,
                confirmed    INTEGER,
                explanation  TEXT,
                key_passage  TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Migrations
        cols = {r[1] for r in conn.execute("PRAGMA table_info(recordings)").fetchall()}
        if 'recording_date' not in cols:
            conn.execute("ALTER TABLE recordings ADD COLUMN recording_date DATE")
        fp_cols = {r[1] for r in conn.execute("PRAGMA table_info(fingerprints)").fetchall()}
        if 'threshold' not in fp_cols:
            conn.execute("ALTER TABLE fingerprints ADD COLUMN threshold REAL")
        if 'sample_count' not in fp_cols:
            conn.execute("ALTER TABLE fingerprints ADD COLUMN sample_count INTEGER DEFAULT 1")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- Fingerprints ---

def save_fingerprint(name: str, embedding: np.ndarray) -> int:
    blob = embedding.astype(np.float32).tobytes()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR REPLACE INTO fingerprints (name, embedding) VALUES (?, ?)",
            (name, blob)
        )
        return cur.lastrowid


def load_fingerprints() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, embedding, threshold, sample_count FROM fingerprints"
        ).fetchall()
    result = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        result.append({
            "id": row["id"], "name": row["name"], "embedding": emb,
            "threshold": row["threshold"],
            "sample_count": row["sample_count"] or 1,
        })
    return result


def list_fingerprints() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, threshold, sample_count, created_at FROM fingerprints ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def update_fingerprint_embedding(fp_id: int, embedding: np.ndarray,
                                 sample_count: int) -> None:
    blob = embedding.astype(np.float32).tobytes()
    with get_conn() as conn:
        conn.execute(
            "UPDATE fingerprints SET embedding = ?, sample_count = ? WHERE id = ?",
            (blob, sample_count, fp_id)
        )


def update_fingerprint_threshold(fp_id: int, threshold: float | None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE fingerprints SET threshold = ? WHERE id = ?",
            (threshold, fp_id)
        )


def get_segments_for_fingerprint(fp_id: int) -> list[dict]:
    """Retourne tous les segments audio d'un locuteur identifié par cette empreinte."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT seg.start_time, seg.end_time, r.filename
               FROM segments seg
               JOIN speakers sp ON seg.speaker_id = sp.id
               JOIN recordings r ON sp.recording_id = r.id
               WHERE sp.fingerprint_id = ?
               ORDER BY r.id, seg.start_time""",
            (fp_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_fingerprint(fp_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM fingerprints WHERE id = ?", (fp_id,))
        return cur.rowcount > 0


# --- Recordings ---

def save_recording(filename: str, duration: float, transcript: str,
                   recording_date=None) -> int:
    date_str = recording_date.isoformat() if recording_date else None
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO recordings (filename, duration, recording_date, transcript_full)"
            " VALUES (?, ?, ?, ?)",
            (filename, duration, date_str, transcript)
        )
        return cur.lastrowid


def get_recordings_without_date() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, transcript_full FROM recordings WHERE recording_date IS NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def update_recording_date(recording_id: int, recording_date) -> None:
    date_str = recording_date.isoformat() if recording_date else None
    with get_conn() as conn:
        conn.execute(
            "UPDATE recordings SET recording_date = ? WHERE id = ?",
            (date_str, recording_id)
        )


def update_recording_summary(recording_id: int, summary: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE recordings SET summary = ? WHERE id = ?",
            (summary, recording_id)
        )


def save_speaker(recording_id: int, label: str, name: str | None,
                 fp_id: int | None, sentiment: str, score: float,
                 speaking_time: float) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO speakers
               (recording_id, speaker_label, identified_name, fingerprint_id,
                overall_sentiment, sentiment_score, speaking_time)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (recording_id, label, name, fp_id, sentiment, score, speaking_time)
        )
        return cur.lastrowid


def save_segments(segments: list[dict]):
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO segments (recording_id, speaker_id, start_time, end_time, text)
               VALUES (:recording_id, :speaker_id, :start, :end, :text)""",
            segments
        )


# --- Consultation ---

def get_recording(recording_id: int) -> dict | None:
    with get_conn() as conn:
        rec = conn.execute(
            "SELECT * FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
        if not rec:
            return None
        speakers = conn.execute(
            "SELECT * FROM speakers WHERE recording_id = ?", (recording_id,)
        ).fetchall()
        segments = conn.execute(
            """SELECT seg.*, sp.speaker_label, sp.identified_name
               FROM segments seg
               JOIN speakers sp ON seg.speaker_id = sp.id
               WHERE seg.recording_id = ?
               ORDER BY seg.start_time""",
            (recording_id,)
        ).fetchall()
    return {
        "recording": dict(rec),
        "speakers": [dict(s) for s in speakers],
        "segments": [dict(s) for s in segments],
    }


def get_analyzed_filenames() -> set[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT filename FROM recordings").fetchall()
    return {row["filename"] for row in rows}


def export_recordings_csv() -> list[dict]:
    with get_conn() as conn:
        recordings = conn.execute(
            "SELECT id, recording_date, duration, transcript_full, summary FROM recordings ORDER BY recording_date, id"
        ).fetchall()
        result = []
        for rec in recordings:
            speakers = conn.execute(
                """SELECT identified_name FROM speakers
                   WHERE recording_id = ? AND fingerprint_id IS NOT NULL
                   ORDER BY speaking_time DESC""",
                (rec["id"],)
            ).fetchall()

            detections = conn.execute(
                """SELECT pd.profile, pd.confirmed, pd.explanation, pd.key_passage,
                          sp.identified_name
                   FROM profile_detections pd
                   LEFT JOIN speakers sp ON pd.speaker_id = sp.id
                   WHERE pd.recording_id = ?""",
                (rec["id"],)
            ).fetchall()

            def _det_cell(profile: str) -> str:
                rows = [d for d in detections if d["profile"] == profile]
                if not rows:
                    return ""
                confirmed_rows = [d for d in rows if d["confirmed"] == 1]
                if not confirmed_rows:
                    return "✗"
                parts = []
                for d in confirmed_rows:
                    who = d["identified_name"] or ""
                    expl = d["explanation"] or ""
                    kp = d["key_passage"] or ""
                    text = f"✓ {expl}"
                    if who:
                        text = f"✓ [{who}] {expl}"
                    if kp:
                        text += f' — « {kp} »'
                    parts.append(text)
                return " // ".join(parts)

            result.append({
                "id": rec["id"],
                "date": rec["recording_date"] or "",
                "duration": rec["duration"] or "",
                "speakers": ", ".join(s["identified_name"] for s in speakers),
                "occupé / activité": _det_cell("busy"),
                "dort chez quelqu'un": _det_cell("sleeping"),
                "qualité audio": _det_cell("quality"),
                "transcription": rec["transcript_full"] or "",
                "résumé": rec["summary"] or "",
            })
    return result


def clear_recordings() -> int:
    with get_conn() as conn:
        conn.execute("DELETE FROM segments")
        conn.execute("DELETE FROM speakers")
        cur = conn.execute("DELETE FROM recordings")
        return cur.rowcount


def save_profile_detection(recording_id: int, speaker_id: int | None, profile: str,
                           confirmed: bool | None, explanation: str | None,
                           key_passage: str | None) -> int:
    conf_int = 1 if confirmed is True else (0 if confirmed is False else None)
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO profile_detections
               (recording_id, speaker_id, profile, confirmed, explanation, key_passage)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (recording_id, speaker_id, profile, conf_int, explanation, key_passage)
        )
        return cur.lastrowid


def get_profile_detections(recording_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT pd.profile, pd.confirmed, pd.explanation, pd.key_passage,
                      sp.speaker_label, sp.identified_name
               FROM profile_detections pd
               LEFT JOIN speakers sp ON pd.speaker_id = sp.id
               WHERE pd.recording_id = ?
               ORDER BY pd.profile, pd.created_at""",
            (recording_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_profile_detections(recording_id: int, profile: str,
                              speaker_name: str | None = None) -> None:
    with get_conn() as conn:
        if speaker_name:
            conn.execute(
                """DELETE FROM profile_detections
                   WHERE recording_id = ? AND profile = ?
                   AND speaker_id IN (
                       SELECT id FROM speakers
                       WHERE recording_id = ? AND identified_name = ?
                   )""",
                (recording_id, profile, recording_id, speaker_name)
            )
        else:
            conn.execute(
                "DELETE FROM profile_detections WHERE recording_id = ? AND profile = ?",
                (recording_id, profile)
            )


def search_segments(keywords: list[str], speaker_filter: str | None = None,
                    recording_id: int | None = None) -> list[dict]:
    """Cherche dans les segments par mots-clés (OR). Retourne les segments qui matchent."""
    if not keywords:
        return []

    like_clauses = " OR ".join("LOWER(seg.text) LIKE ?" for _ in keywords)
    params: list = [f"%{kw.lower()}%" for kw in keywords]

    speaker_clause = ""
    if speaker_filter:
        speaker_clause = " AND (LOWER(sp.identified_name) LIKE ? OR LOWER(sp.speaker_label) LIKE ?)"
        sf = f"%{speaker_filter.lower()}%"
        params += [sf, sf]

    rec_clause = ""
    if recording_id is not None:
        rec_clause = " AND r.id = ?"
        params.append(recording_id)

    query = f"""
        SELECT seg.id, seg.start_time, seg.end_time, seg.text,
               sp.speaker_label, sp.identified_name,
               r.id as recording_id, r.filename, r.recording_date
        FROM segments seg
        JOIN speakers sp ON seg.speaker_id = sp.id
        JOIN recordings r ON seg.recording_id = r.id
        WHERE ({like_clauses}){speaker_clause}{rec_clause}
        ORDER BY r.recording_date DESC, r.id DESC, seg.start_time
    """
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def list_recordings() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT r.id, r.filename, r.duration, r.recording_date, r.created_at,
                      COUNT(DISTINCT s.id) as speaker_count
               FROM recordings r
               LEFT JOIN speakers s ON s.recording_id = r.id
               GROUP BY r.id
               ORDER BY r.created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]
