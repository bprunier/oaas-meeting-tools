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
        # Migration base existante sans la colonne recording_date
        cols = {r[1] for r in conn.execute("PRAGMA table_info(recordings)").fetchall()}
        if 'recording_date' not in cols:
            conn.execute("ALTER TABLE recordings ADD COLUMN recording_date DATE")


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
        rows = conn.execute("SELECT id, name, embedding FROM fingerprints").fetchall()
    result = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        result.append({"id": row["id"], "name": row["name"], "embedding": emb})
    return result


def list_fingerprints() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM fingerprints ORDER BY name"
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
            result.append({
                "id": rec["id"],
                "date": rec["recording_date"] or "",
                "duration": rec["duration"] or "",
                "speakers": ", ".join(s["identified_name"] for s in speakers),
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
