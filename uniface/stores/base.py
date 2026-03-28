# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = ['BaseStore']

Metadata = dict[str, Any]


class BaseStore(ABC):
    """Abstract interface for face embedding vector stores.

    All vector store backends (FAISS, Qdrant, etc.) must implement
    this interface to ensure consistent usage across the library.

    Embeddings are expected to be L2-normalised so that inner product
    equals cosine similarity.
    """

    @abstractmethod
    def add(self, embedding: np.ndarray, metadata: Metadata) -> None:
        """Add a single embedding with associated metadata.

        Args:
            embedding: L2-normalised embedding vector.
            metadata: Arbitrary dict of JSON-serialisable key-value pairs.
        """

    @abstractmethod
    def search(
        self,
        embedding: np.ndarray,
        threshold: float = 0.4,
    ) -> tuple[Metadata | None, float]:
        """Find the closest match for a query embedding.

        Args:
            embedding: L2-normalised query vector.
            threshold: Minimum similarity to accept a match.

        Returns:
            ``(metadata, similarity)`` for the best match, or
            ``(None, similarity)`` when below *threshold* or empty.
        """

    @abstractmethod
    def remove(self, key: str, value: Any) -> int:
        """Remove all entries where ``metadata[key] == value``.

        Args:
            key: Metadata key to match against.
            value: Value to match.

        Returns:
            Number of entries removed.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of vectors in the store."""
