from __future__ import annotations
from datetime import date, datetime, timedelta, timezone


def _escape(text: str) -> str:
    text = text.replace('\\', '\\\\')
    text = text.replace(';', '\\;')
    text = text.replace(',', '\\,')
    text = text.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
    return text


def _fold(line: str) -> str:
    """Repliage RFC 5545 : max 75 octets par ligne, continuation avec CRLF + espace."""
    result = []
    encoded = line.encode('utf-8')
    while len(encoded) > 75:
        cut = 75
        # Ne pas couper au milieu d'un caractère multi-octet
        while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
            cut -= 1
        result.append(encoded[:cut].decode('utf-8'))
        encoded = b' ' + encoded[cut:]
    result.append(encoded.decode('utf-8'))
    return '\r\n'.join(result)


def _ambiance(speakers: list[dict]) -> str:
    if not speakers:
        return "Inconnue"
    scores = [s.get('sentiment_score') or 0.0 for s in speakers]
    avg = sum(scores) / len(scores)
    if avg >= 0.3:
        return "Positive"
    if avg <= -0.3:
        return "Négative"
    return "Neutre"


def _participants(speakers: list[dict]) -> str:
    names = [s.get('identified_name') or s['speaker_label'] for s in speakers]
    return ', '.join(names)


def _vevent(rec_data: dict, dtstamp: str) -> list[str]:
    rec = rec_data['recording']
    speakers = rec_data['speakers']

    raw_date = rec.get('recording_date')
    if raw_date:
        d = date.fromisoformat(raw_date)
    else:
        d = datetime.fromisoformat(rec['created_at'][:10]).date()

    ambiance = _ambiance(speakers)
    participants = _participants(speakers)
    summary_line = f"Réunion {ambiance} – {participants}" if participants else f"Réunion {ambiance}"
    description = rec.get('summary') or ''

    return [
        'BEGIN:VEVENT',
        f'UID:recording-{rec["id"]}@audio-analyzer',
        f'DTSTAMP:{dtstamp}',
        f'DTSTART;VALUE=DATE:{d.strftime("%Y%m%d")}',
        f'DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime("%Y%m%d")}',
        _fold(f'SUMMARY:{_escape(summary_line)}'),
        _fold(f'DESCRIPTION:{_escape(description)}'),
        'END:VEVENT',
    ]


def build_ics(recordings_data: list[dict]) -> str:
    dtstamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    lines = [
        'BEGIN:VCALENDAR',
        'VERSION:2.0',
        'PRODID:-//Audio Analyzer//FR',
        'CALSCALE:GREGORIAN',
        'METHOD:PUBLISH',
    ]
    for rec_data in recordings_data:
        lines.extend(_vevent(rec_data, dtstamp))
    lines.append('END:VCALENDAR')
    return '\r\n'.join(lines) + '\r\n'
