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


# --- Recordings ---

def save_recording(filename: str, duration: float, transcript: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO recordings (filename, duration, transcript_full) VALUES (?, ?, ?)",
            (filename, duration, transcript)
        )
        return cur.lastrowid


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


def list_recordings() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT r.id, r.filename, r.duration, r.created_at,
                      COUNT(DISTINCT s.id) as speaker_count
               FROM recordings r
               LEFT JOIN speakers s ON s.recording_id = r.id
               GROUP BY r.id
               ORDER BY r.created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]
