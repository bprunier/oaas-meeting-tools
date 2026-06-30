"""
Gestion des projets : isolation complète des données par projet.
Chaque projet possède sa propre BDD SQLite, ChromaDB, répertoire voices et exports.
"""
from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

PROJECTS_DIR = Path("projects")
CURRENT_PROJECT_FILE = Path(".current_project")


def project_path(name: str) -> Path:
    return PROJECTS_DIR / name


def create_project(name: str) -> Path:
    """Crée l'arborescence d'un nouveau projet."""
    p = project_path(name)
    (p / "voices").mkdir(parents=True, exist_ok=True)
    (p / "exports").mkdir(exist_ok=True)
    (p / "chroma_db").mkdir(exist_ok=True)

    env_path = p / ".env"
    if not env_path.exists():
        env_path.write_text(
            f"# Projet : {name}\n"
            "# Surcharge les paramètres du .env global (Ollama, Whisper, etc.)\n\n"
            "# WHISPER_MODEL_SIZE=large-v3\n"
            "# AUDIO_LANGUAGE=fr\n"
            "# OLLAMA_MODEL=llama3\n"
            "# EMBEDDING_MODEL=nomic-embed-text\n"
            "# MIN_SPEAKING_TIME=5\n",
            encoding="utf-8",
        )
    return p


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(d.name for d in PROJECTS_DIR.iterdir() if d.is_dir())


def get_current_project() -> str | None:
    if CURRENT_PROJECT_FILE.exists():
        name = CURRENT_PROJECT_FILE.read_text(encoding="utf-8").strip()
        return name or None
    return None


def set_current_project(name: str | None) -> None:
    if name is None:
        CURRENT_PROJECT_FILE.unlink(missing_ok=True)
    else:
        CURRENT_PROJECT_FILE.write_text(name, encoding="utf-8")


def activate_project(name: str) -> Path:
    """
    Résout les chemins du projet et met à jour config.
    À appeler avant db.init_db() et toute autre opération.
    """
    from audio_analyzer import config, embedder

    p = project_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"Projet '{name}' introuvable. "
            f"Créez-le avec : python main.py project create {name}"
        )

    # Charger le .env spécifique au projet (surcharge le .env global)
    env_file = p / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)

    # Chemins toujours dérivés du répertoire du projet (non surchargeables)
    config.DB_PATH = str(p / "audio_analysis.db")
    config.CHROMA_PATH = str(p / "chroma_db")

    # Réinitialiser le singleton ChromaDB (chemin a changé)
    embedder._collection = None

    return p


def exports_dir(name: str) -> Path:
    return project_path(name) / "exports"


def voices_dir(name: str) -> Path:
    return project_path(name) / "voices"
