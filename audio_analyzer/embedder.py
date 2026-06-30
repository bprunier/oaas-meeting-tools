"""
Embeddings sémantiques (sentence-transformers) + stockage vectoriel ChromaDB local.
Permet la recherche sémantique sur les segments de transcription.
"""
from __future__ import annotations
import torch
from audio_analyzer.config import EMBEDDING_MODEL, CHROMA_PATH

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Chargement embeddings [{device}]: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name="segments",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return vectors.tolist()


def index_segments(segments: list[dict], recording_id: int,
                   filename: str, recording_date=None) -> int:
    """
    Indexe une liste de segments dans ChromaDB.
    segments doit contenir : id, text, speaker_label, identified_name, start_time, end_time
    """
    if not segments:
        return 0

    texts = [s["text"] for s in segments]
    embeddings = embed_texts(texts)

    ids = [str(s["id"]) for s in segments]
    metadatas = [
        {
            "recording_id": int(recording_id),
            "filename": str(filename),
            "recording_date": str(recording_date or ""),
            "speaker_label": str(s.get("speaker_label") or ""),
            "identified_name": str(s.get("identified_name") or ""),
            "start_time": float(s.get("start_time", 0)),
            "end_time": float(s.get("end_time", 0)),
        }
        for s in segments
    ]

    _get_collection().upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(segments)


def search_semantic(query: str, n_results: int = 10,
                    recording_id: int | None = None) -> list[dict]:
    """
    Recherche sémantique dans les segments indexés.
    Retourne une liste de segments triés par score de similarité décroissant.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_embedding = embed_texts([query])[0]
    n = min(n_results, count)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if recording_id is not None:
        kwargs["where"] = {"recording_id": {"$eq": int(recording_id)}}

    results = collection.query(**kwargs)

    segments = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        segments.append({
            "text": doc,
            "score": round(1.0 - distance, 4),
            "recording_id": meta["recording_id"],
            "filename": meta["filename"],
            "recording_date": meta["recording_date"],
            "speaker_label": meta["speaker_label"],
            "identified_name": meta["identified_name"],
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
        })

    return sorted(segments, key=lambda x: x["score"], reverse=True)


def collection_count() -> int:
    return _get_collection().count()
