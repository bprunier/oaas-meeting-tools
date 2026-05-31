"""
Recherche dans les transcripts : mots-clés + confirmation Ollama.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from audio_analyzer import database as db
from audio_analyzer.analyzer import _chat

PROFILES: dict[str, dict] = {
    "busy": {
        "label": "Interlocuteur occupé / activité / ami présent",
        "keywords": [
            # Indisponibilité directe
            "occupé", "occupée", "pas disponible", "j'ai pas le temps",
            "je peux pas", "indisponible", "chargé", "surchargé", "débordé",
            "je suis pris", "je suis prise", "pas le temps", "je ne peux pas",
            # Ami / visite présente
            "je suis avec", "il est avec moi", "elle est avec moi",
            "j'ai de la visite", "j'ai quelqu'un", "j'ai des invités",
            "copain", "copine", "il vient me voir", "elle vient me voir",
            # En train de faire une activité
            "je joue", "on joue", "je regarde", "on regarde",
            "je suis en train", "on est en train",
            "match", "film", "série", "épisode",
            "je sors", "on sort", "promenade", "balade",
            "je cuisine", "on mange", "je travaille",
        ],
        "ollama_description": (
            "Un participant est occupé ou indisponible. Cela inclut : "
            "(1) il signale explicitement être occupé ou ne pas avoir le temps, "
            "(2) il est en train de faire une activité (regarder un film, jouer, sortir, manger…), "
            "(3) il a quelqu'un avec lui (ami, visite, copain/copine). "
            "Indique dans l'explication laquelle de ces trois situations s'applique."
        ),
    },
    "sleeping": {
        "label": "Dormir chez quelqu'un",
        "keywords": [
            "dormir chez", "je dors chez", "passer la nuit", "coucher chez",
            "héberge", "je reste chez", "nuit chez", "aller dormir",
            "je vais dormir", "dormir là", "dormir là-bas",
        ],
        "ollama_description": (
            "Un participant mentionne qu'il va dormir ou passer la nuit "
            "chez quelqu'un d'autre."
        ),
    },
    "quality": {
        "label": "Problème de qualité audio",
        "keywords": [
            "ça coupe", "tu m'entends", "j'entends pas", "je t'entends pas",
            "tu entends", "ça lag", "mauvais son", "coupure",
            "répète", "problème de son", "micro", "ça freeze",
            "tu casses", "ça saute", "connexion", "signal", "bruit",
            "j'entends rien", "on s'entend pas",
        ],
        "ollama_description": (
            "Il y a des problèmes de qualité audio, de connexion "
            "ou d'audibilité durant la conversation."
        ),
    },
}


def _build_ollama_prompt(filename: str, date: str | None,
                         segments: list[dict], pattern_description: str) -> str:
    excerpt = "\n".join(
        f"  [{_fmt(seg['start_time'])}] "
        f"{seg.get('identified_name') or seg['speaker_label']}: {seg['text']}"
        for seg in segments
    )
    date_str = date or "date inconnue"
    return f"""Tu es un assistant qui analyse des transcripts de réunions audio.

Pattern recherché : {pattern_description}

Extrait de transcript (fichier "{filename}", {date_str}) :
{excerpt}

Réponds UNIQUEMENT en JSON valide, sans texte autour :
{{
  "confirmed": true | false,
  "explanation": "explication courte en français (1-2 phrases)",
  "key_passage": "citation exacte la plus pertinente, ou null si non confirmé"
}}"""


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def confirm_profile(segments: list[dict], profile_name: str,
                    filename: str = "", date: str | None = None) -> dict:
    """
    Détecte un profil sur des segments en mémoire (pendant l'analyse).
    segments : format transcriber {"speaker", "text", "start", "end"}
    """
    if profile_name not in PROFILES:
        raise ValueError(f"Profil inconnu : {profile_name!r}")
    prof = PROFILES[profile_name]
    keywords = prof["keywords"]

    matched = [
        seg for seg in segments
        if any(kw.lower() in seg.get("text", "").lower() for kw in keywords)
    ]
    if not matched:
        return {"confirmed": False, "explanation": "Aucun mot-clé détecté.", "key_passage": None}

    # Normaliser vers le format attendu par _confirm (champs DB)
    normalized = [
        {
            "start_time": seg.get("start", seg.get("start_time", 0)),
            "identified_name": seg.get("identified_name"),
            "speaker_label": seg.get("speaker", seg.get("speaker_label", "?")),
            "text": seg.get("text", ""),
        }
        for seg in matched
    ]
    return _confirm(filename, date, normalized, prof["ollama_description"])


def search(
    query: str | None,
    profile: str | None,
    speaker_filter: str | None = None,
    confirm: bool = True,
    recording_id: int | None = None,
) -> list[dict]:
    """
    Retourne une liste de résultats groupés par enregistrement :
    [
      {
        "recording": {...},
        "matches": [segment_dict, ...],
        "ollama": {"confirmed": bool, "explanation": str, "key_passage": str|None} | None,
      },
      ...
    ]
    """
    if profile:
        if profile not in PROFILES:
            raise ValueError(f"Profil inconnu : {profile!r}. Disponibles : {list(PROFILES)}")
        keywords = PROFILES[profile]["keywords"]
        description = PROFILES[profile]["ollama_description"]
    elif query:
        keywords = [query]
        description = f'Quelqu\'un dit quelque chose contenant "{query}".'
    else:
        raise ValueError("Fournir --query ou --profile.")

    raw_matches = db.search_segments(keywords, speaker_filter, recording_id)
    if not raw_matches:
        return []

    # Grouper par enregistrement
    by_rec: dict[int, dict] = {}
    for seg in raw_matches:
        rid = seg["recording_id"]
        if rid not in by_rec:
            by_rec[rid] = {
                "recording": {
                    "id": rid,
                    "filename": seg["filename"],
                    "recording_date": seg["recording_date"],
                },
                "matches": [],
                "ollama": None,
            }
        by_rec[rid]["matches"].append(seg)

    results = list(by_rec.values())

    if confirm:
        for entry in results:
            rec = entry["recording"]
            entry["ollama"] = _confirm(
                filename=Path(rec["filename"]).name,
                date=rec["recording_date"],
                segments=entry["matches"],
                description=description,
            )

    return results


def _parse_json_robust(raw: str) -> dict:
    """Extrait du JSON depuis une réponse LLM potentiellement tronquée ou enrobée."""
    # 1. Nettoyer les balises code markdown
    text = raw.strip()
    if "```" in text:
        inner = text.split("```")[1]
        text = inner[4:].strip() if inner.startswith("json") else inner.strip()

    # 2. Tenter parse direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Extraire entre le premier { et le dernier }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 4. JSON tronqué : tenter de refermer les accolades manquantes
    if start != -1:
        fragment = text[start:]
        missing = fragment.count("{") - fragment.count("}")
        for suffix in [("}" * max(1, missing)), '"}' + "}" * max(0, missing - 1)]:
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("Impossible d'extraire le JSON", raw, 0)


def _confirm(filename: str, date: str | None, segments: list[dict],
             description: str) -> dict:
    prompt = _build_ollama_prompt(filename, date, segments, description)
    raw = _chat(prompt)
    try:
        return _parse_json_robust(raw)
    except json.JSONDecodeError:
        return {"confirmed": None, "explanation": raw[:200], "key_passage": None}
