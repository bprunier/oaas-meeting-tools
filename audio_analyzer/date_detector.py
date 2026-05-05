from __future__ import annotations
import re
from datetime import date, datetime
from pathlib import Path

_MONTHS = {
    # Français
    'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4,
    'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8, 'aout': 8,
    'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12, 'decembre': 12,
    # English
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12,
}


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _from_filename(path: Path) -> date | None:
    stem = path.stem
    # YYYY-MM-DD ou YYYY_MM_DD
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', stem)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    # YYYYMMDDHHMMSS (enregistreurs : 20250914175924)
    m = re.search(r'(?<!\d)(\d{4})(\d{2})(\d{2})\d{6}(?!\d)', stem)
    if m:
        d = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        if d:
            return d
    # YYYYMMDD (8 chiffres isolés)
    m = re.search(r'(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)', stem)
    if m:
        d = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        if d:
            return d
    # DD-MM-YYYY ou DD/MM/YYYY
    m = re.search(r'(\d{2})[-_/](\d{2})[-_/](\d{4})', stem)
    if m:
        return _safe_date(int(m[3]), int(m[2]), int(m[1]))
    return None


def _from_transcript(text: str) -> date | None:
    if not text:
        return None
    # ISO : 2024-01-15
    m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', text)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    # DD/MM/YYYY ou DD-MM-YYYY
    m = re.search(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b', text)
    if m:
        return _safe_date(int(m[3]), int(m[2]), int(m[1]))
    # "15 janvier 2024" / "15 january 2024"
    month_pat = '|'.join(re.escape(k) for k in _MONTHS)
    m = re.search(rf'\b(\d{{1,2}})\s+({month_pat})\s+(\d{{4}})\b', text, re.IGNORECASE)
    if m:
        month = _MONTHS.get(m[2].lower())
        if month:
            return _safe_date(int(m[3]), month, int(m[1]))
    return None


def _from_metadata(path: Path) -> date | None:
    stat = path.stat()
    # st_birthtime sur macOS, st_ctime sur Windows (date de création)
    ts = getattr(stat, 'st_birthtime', None) or stat.st_ctime
    return datetime.fromtimestamp(ts).date()


def detect_recording_date(audio_path: str, transcript: str = "") -> date | None:
    """Détecte la date de l'enregistrement : nom de fichier > transcript > métadonnées."""
    path = Path(audio_path)
    return (
        _from_filename(path)
        or _from_transcript(transcript)
        or _from_metadata(path)
    )
