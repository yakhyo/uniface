# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from uniface.log import Logger

__all__ = ['FAISS']

Metadata = dict[str, Any]


def _import_faiss():
    """Lazily import faiss, raising a clear error if not installed."""
    # Prevent OpenMP abort on macOS when multiple libraries (e.g. scipy,
    # torch) each bundle their own libomp.
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            'faiss is required for FAISS vector store. '
            'Install it with:  pip install faiss-cpu  (CPU) '
            'or:  pip install faiss-gpu  (CUDA)'
        ) from exc
    return faiss


class FAISS:
    """FAISS vector store using IndexFlatIP (inner product).

    Vectors must be L2-normalised **before** being added so that inner
    product equals cosine similarity.  The store does not normalise
    internally -- that is the caller's responsibility.

    Each vector is paired with a metadata dict that can carry any
    JSON-serialisable payload (person ID, name, source image, etc.).

    Args:
        embedding_size: Dimension of embedding vectors.
        db_path: Directory for persisting the index and metadata.

    Example:
        >>> from uniface.indexing import FAISS
        >>> store = FAISS(embedding_size=512, db_path='./my_index')
        >>> store.add(embedding, {'person_id': '001', 'name': 'Alice'})
        >>> result, score = store.search(query_embedding)
        >>> result['name']
        'Alice'
    """

    def __init__(
        self,
        embedding_size: int = 512,
        db_path: str = './vector_index',
    ) -> None:
        faiss = _import_faiss()

        self.embedding_size = embedding_size
        self.db_path = db_path
        self._index_file = os.path.join(db_path, 'faiss_index.bin')
        self._meta_file = os.path.join(db_path, 'metadata.json')

        os.makedirs(db_path, exist_ok=True)

        self.index = faiss.IndexFlatIP(embedding_size)
        self.metadata: list[Metadata] = []

    def add(self, embedding: np.ndarray, metadata: Metadata) -> None:
        """Add a single embedding with associated metadata.

        Args:
            embedding: Embedding vector (must be L2-normalised).
            metadata: Arbitrary dict of JSON-serialisable key-value pairs.
        """
        vec = self._prepare(embedding).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append(metadata)

    def search(
        self,
        embedding: np.ndarray,
        threshold: float = 0.4,
    ) -> tuple[Metadata | None, float]:
        """Find the closest match for a query embedding.

        Args:
            embedding: Query embedding vector (must be L2-normalised).
            threshold: Minimum cosine similarity to accept a match.

        Returns:
            ``(metadata, similarity)`` for the best match, or
            ``(None, similarity)`` when below *threshold* or the
            index is empty.
        """
        if self.index.ntotal == 0:
            return None, 0.0

        vec = self._prepare(embedding).reshape(1, -1)
        similarities, indices = self.index.search(vec, 1)

        similarity = float(similarities[0][0])
        idx = int(indices[0][0])

        if similarity > threshold and 0 <= idx < len(self.metadata):
            return self.metadata[idx], similarity
        return None, similarity

    def remove(self, key: str, value: Any) -> int:
        """Remove all entries where ``metadata[key] == value`` and rebuild.

        Args:
            key: Metadata key to match against.
            value: Value to match.

        Returns:
            Number of entries removed.
        """
        faiss = _import_faiss()

        keep = [i for i, m in enumerate(self.metadata) if m.get(key) != value]
        removed = len(self.metadata) - len(keep)
        if removed == 0:
            return 0

        if keep:
            vectors = np.empty((len(keep), self.embedding_size), dtype=np.float32)
            for dst, src in enumerate(keep):
                self.index.reconstruct(src, vectors[dst])
            new_index = faiss.IndexFlatIP(self.embedding_size)
            new_index.add(vectors)
        else:
            new_index = faiss.IndexFlatIP(self.embedding_size)

        self.index = new_index
        self.metadata = [self.metadata[i] for i in keep]
        Logger.info('Removed %d entries where %s=%s (%d remaining)', removed, key, value, self.index.ntotal)
        return removed

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        faiss = _import_faiss()

        faiss.write_index(self.index, self._index_file)
        with open(self._meta_file, 'w', encoding='utf-8') as fh:
            json.dump(self.metadata, fh, ensure_ascii=False, indent=2)
        Logger.info('Saved FAISS index with %d vectors to %s', self.index.ntotal, self.db_path)

    def load(self) -> bool:
        """Load a previously saved index and metadata from disk.

        Returns:
            ``True`` if loaded successfully, ``False`` if files are missing.

        Raises:
            RuntimeError: If files exist but cannot be read.
        """
        if not (os.path.exists(self._index_file) and os.path.exists(self._meta_file)):
            return False

        faiss = _import_faiss()

        try:
            loaded_index = faiss.read_index(self._index_file)
            with open(self._meta_file, encoding='utf-8') as fh:
                loaded_metadata: list[Metadata] = json.load(fh)
        except Exception as exc:
            raise RuntimeError(f'Failed to load FAISS index from {self.db_path}') from exc

        self.index = loaded_index
        self.metadata = loaded_metadata
        Logger.info('Loaded FAISS index with %d vectors from %s', self.index.ntotal, self.db_path)
        return True

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    @staticmethod
    def _prepare(vec: np.ndarray) -> np.ndarray:
        """Cast to contiguous float32 for FAISS compatibility."""
        return np.ascontiguousarray(vec.ravel(), dtype=np.float32)

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return f'FAISS(embedding_size={self.embedding_size}, vectors={self.index.ntotal})'
