# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Vector store backends for fast face embedding similarity search."""

from uniface.stores.base import BaseStore
from uniface.stores.faiss import FAISS

__all__ = ['BaseStore', 'FAISS']
