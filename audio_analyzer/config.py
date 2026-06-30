import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_PATH = os.getenv("DB_PATH", "audio_analysis.db")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
AUDIO_LANGUAGE = os.getenv("AUDIO_LANGUAGE") or None  # None = auto-détection

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

MIN_SPEAKING_TIME = float(os.getenv("MIN_SPEAKING_TIME", "5"))
VAD_TOP_DB = int(os.getenv("VAD_TOP_DB", "35"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))


def validate():
    pass  # Tout est local, aucune clé requise
