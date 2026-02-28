"""Build and query FAISS vector index from processed chunks."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "meta.pkl"


class VectorStore:
    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path or settings.vector_store_path)
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: List[Dict] = []

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            log.info("loading_embedding_model", model=settings.embedding_model)
            self.model = SentenceTransformer(settings.embedding_model)
        return self.model

    def build(self, chunks_parquet: Optional[str] = None) -> None:
        """Embed all chunks and write FAISS index + metadata."""
        parquet_path = chunks_parquet or str(Path(settings.processed_data_dir) / "chunks.parquet")
        df = pd.read_parquet(parquet_path)
        log.info("building_index", chunks=len(df))

        model = self._get_model()
        texts = df["text"].tolist()
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.metadata = df[["id", "document_id", "text", "title", "url", "date_published", "category"]].to_dict("records")

        self.store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.store_path / _INDEX_FILE))
        with open(self.store_path / _META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

        log.info("index_built", vectors=self.index.ntotal, path=str(self.store_path))

    def load(self) -> None:
        """Load existing index from disk."""
        idx_path = self.store_path / _INDEX_FILE
        meta_path = self.store_path / _META_FILE
        if not idx_path.exists():
            raise FileNotFoundError(f"No FAISS index found at {idx_path}. Run ingest first.")
        self.index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        log.info("index_loaded", vectors=self.index.ntotal)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k chunks for a query."""
        if self.index is None:
            self.load()
        model = self._get_model()
        q_emb = model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)
        return results


# Singleton
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store