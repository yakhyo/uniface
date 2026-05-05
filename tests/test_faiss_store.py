"""Tests for FAISS vector store, focused on the remove() refactor."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.stores.faiss import FAISS

faiss = pytest.importorskip('faiss')


def _unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


@pytest.fixture
def store(tmp_path):
    return FAISS(embedding_size=16, db_path=str(tmp_path / 'idx'))


def _populate(store: FAISS, n: int = 6, seed: int = 1) -> np.ndarray:
    vecs = _unit_vectors(n, store.embedding_size, seed=seed)
    for i, v in enumerate(vecs):
        store.add(v, {'person_id': f'p{i}', 'group': 'odd' if i % 2 else 'even'})
    return vecs


def test_remove_no_match_is_noop(store):
    _populate(store, n=4)
    assert store.remove('person_id', 'does-not-exist') == 0
    assert store.size == 4
    assert len(store.metadata) == 4


def test_remove_single_entry_preserves_alignment(store):
    vecs = _populate(store, n=6)
    removed = store.remove('person_id', 'p2')
    assert removed == 1
    assert store.size == 5
    assert len(store.metadata) == 5
    assert [m['person_id'] for m in store.metadata] == ['p0', 'p1', 'p3', 'p4', 'p5']

    # Each surviving metadata still maps to the correct vector
    for new_pos, m in enumerate(store.metadata):
        original = int(m['person_id'][1:])
        out = np.empty(store.embedding_size, dtype=np.float32)
        store.index.reconstruct(new_pos, out)
        assert np.allclose(out, vecs[original]), f'misaligned at new_pos={new_pos}'


def test_remove_multiple_entries_by_group(store):
    _populate(store, n=6)  # p0,p2,p4 are 'even'; p1,p3,p5 are 'odd'
    removed = store.remove('group', 'odd')
    assert removed == 3
    assert store.size == 3
    assert {m['person_id'] for m in store.metadata} == {'p0', 'p2', 'p4'}
    assert all(m['group'] == 'even' for m in store.metadata)


def test_remove_all_entries(store):
    _populate(store, n=4)
    # All four share key 'group' with one of two values; remove both.
    store.remove('group', 'even')
    store.remove('group', 'odd')
    assert store.size == 0
    assert store.metadata == []
    # search on an empty index returns the documented (None, 0.0)
    q = _unit_vectors(1, store.embedding_size, seed=99)[0]
    assert store.search(q) == (None, 0.0)


def test_search_after_remove_matches_fresh_index(store, tmp_path):
    vecs = _populate(store, n=20, seed=3)
    store.remove('person_id', 'p7')
    store.remove('person_id', 'p13')

    # Build a reference store containing only the surviving vectors in the
    # same order, and confirm the top match is identical for several queries.
    ref = FAISS(embedding_size=store.embedding_size, db_path=str(tmp_path / 'ref'))
    for i, v in enumerate(vecs):
        if i in (7, 13):
            continue
        ref.add(v, {'person_id': f'p{i}', 'group': 'odd' if i % 2 else 'even'})

    queries = _unit_vectors(5, store.embedding_size, seed=42)
    for q in queries:
        m_a, s_a = store.search(q, threshold=-1.0)  # threshold below all sims
        m_b, s_b = ref.search(q, threshold=-1.0)
        assert m_a == m_b
        assert s_a == pytest.approx(s_b, abs=1e-5)


def test_remove_then_save_load_roundtrip(tmp_path):
    db_path = str(tmp_path / 'persist')
    store = FAISS(embedding_size=16, db_path=db_path)
    _populate(store, n=5, seed=7)
    store.remove('person_id', 'p1')
    store.remove('person_id', 'p3')
    store.save()

    reopened = FAISS(embedding_size=16, db_path=db_path)
    assert reopened.load() is True
    assert reopened.size == 3
    assert [m['person_id'] for m in reopened.metadata] == ['p0', 'p2', 'p4']

    # Searching for the original vector of a survivor still finds it
    q_vecs = _unit_vectors(5, 16, seed=7)
    match, sim = reopened.search(q_vecs[2], threshold=0.5)
    assert match is not None
    assert match['person_id'] == 'p2'
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_add_after_remove_keeps_alignment(store):
    _populate(store, n=4)
    store.remove('person_id', 'p1')
    new_vec = _unit_vectors(1, store.embedding_size, seed=123)[0]
    store.add(new_vec, {'person_id': 'p_new'})
    assert store.size == 4
    assert [m['person_id'] for m in store.metadata] == ['p0', 'p2', 'p3', 'p_new']

    match, sim = store.search(new_vec, threshold=0.5)
    assert match is not None
    assert match['person_id'] == 'p_new'
    assert sim == pytest.approx(1.0, abs=1e-5)
