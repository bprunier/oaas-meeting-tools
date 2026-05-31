"""
Analyse de sentiment par locuteur et résumé global via Ollama (LLM local).
"""
from __future__ import annotations
import json
import ollama
from audio_analyzer.config import OLLAMA_HOST, OLLAMA_MODEL


def _chat(prompt: str) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def _extract_json(raw: str) -> dict:
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def analyze_sentiments(speakers_texts: dict[str, str]) -> dict[str, dict]:
    """
    speakers_texts : {speaker_label: texte_complet}
    Retourne : {speaker_label: {sentiment, score, explication}}
    """
    formatted = "\n\n".join(
        f"### {label}\n{text}" for label, text in speakers_texts.items()
    )

    prompt = f"""Tu es un expert en analyse de sentiment appliquée aux conversations.

Voici les transcriptions de chaque locuteur dans une réunion ou conversation :

{formatted}

Pour chaque locuteur, analyse le sentiment global de ses propos.
Réponds UNIQUEMENT en JSON valide avec ce format exact, sans texte autour :
{{
  "SPEAKER_00": {{
    "sentiment": "positif" | "négatif" | "neutre" | "mitigé",
    "score": <float entre -1.0 (très négatif) et 1.0 (très positif)>,
    "explication": "<une phrase résumant le ton général>"
  }}
}}"""

    raw = _chat(prompt)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        # Fallback : retourner neutre pour chaque locuteur si le modèle ne respecte pas le format
        return {
            label: {"sentiment": "neutre", "score": 0.0, "explication": "Analyse indisponible."}
            for label in speakers_texts
        }


_PROFILE_LABELS = {
    "busy": "Occupé(e) / Indisponible",
    "sleeping": "Dort chez quelqu'un",
    "quality": "Problème qualité audio",
}


def generate_summary(transcript: str, speakers_info: dict[str, dict],
                     participants: dict[str, str] | None = None,
                     detections: list[dict] | None = None) -> str:
    """
    transcript    : texte complet de la réunion
    speakers_info : {label: {sentiment, score, ...}}
    participants  : {label: nom_affiché} — nom identifié ou label brut
    detections    : liste de dicts {profile, confirmed, explanation, key_passage, identified_name, speaker_label}
    """
    sentiments_str = "\n".join(
        f"- {(participants or {}).get(label, label)} ({info.get('sentiment', '?')}) : {info.get('explication', '')}"
        for label, info in speakers_info.items()
    )

    if participants:
        names = ", ".join(participants.values())
        participants_line = f"Participants ({len(participants)}) : {names}\n\n"
    else:
        participants_line = ""

    detections_block = ""
    if detections:
        confirmed = [d for d in detections if d.get("confirmed")]
        if confirmed:
            lines = []
            for d in confirmed:
                who = d.get("identified_name") or d.get("speaker_label") or "enregistrement"
                label = _PROFILE_LABELS.get(d["profile"], d["profile"])
                line = f"- {label} ({who})"
                if d.get("key_passage"):
                    line += f' : « {d["key_passage"]} »'
                lines.append(line)
            detections_block = (
                "\nSignaux détectés automatiquement :\n"
                + "\n".join(lines)
                + "\n"
            )

    prompt = f"""Tu es un assistant expert en synthèse de réunions.

Voici la transcription complète d'une réunion :

{transcript}

{participants_line}Sentiments identifiés par locuteur :
{sentiments_str}
{detections_block}
Génère un résumé structuré en français comprenant :
1. **Contexte** – Sujet principal de la réunion (2-3 phrases)
2. **Participants** – Nombre et noms des participants (utilise les noms fournis si disponibles)
3. **Points clés abordés** – Liste des sujets importants
4. **Décisions prises** – Ce qui a été décidé (si applicable)
5. **Actions à suivre** – Tâches identifiées avec responsable si mentionné
6. **Ambiance générale** – Dynamique du groupe et tensions éventuelles
{("7. **Signaux détectés** – Reprends les signaux ci-dessus dans le résumé (disponibilité, hébergement, qualité audio)" if detections_block else "")}

Sois précis et factuel."""

    return _chat(prompt)
