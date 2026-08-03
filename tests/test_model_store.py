# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

from enum import Enum
import os

import pytest

from uniface import model_store
import uniface.constants as const
from uniface.model_store import _mirror_url, verify_model_weights


@pytest.fixture
def model():
    """A registry entry to exercise source resolution against."""
    return const.BlazeFaceWeights.DEFAULT


class TestMirrorUrl:
    """Mirror URL construction."""

    def test_mirror_url_shape(self, model):
        """Assert the literal URL, so a wrong repo id or revision fails the test."""
        assert _mirror_url(model, const.MODEL_REGISTRY[model].url) == (
            'https://huggingface.co/yakhyo/uniface-weights/resolve/'
            '4c7ed723a20deb7ff154b1ba7d6e73747d954016/blazeface.onnx'
        )

    def test_extension_follows_primary_url(self, model):
        """Not every weight is ONNX; the mirror must not hardcode the extension."""
        assert _mirror_url(model, 'https://example.com/x/affecnet7.script').endswith(f'/{model.value}.script')

    def test_mirror_names_unique_across_registry(self):
        """A basename-keyed mirror would collide; enum values are the only unique key."""
        mirrors = [_mirror_url(k, v.url) for k, v in const.MODEL_REGISTRY.items()]
        assert len(set(mirrors)) == len(const.MODEL_REGISTRY)


class TestFallback:
    """Failover behaviour across sources, with no network access."""

    def test_falls_back_to_second_source(self, model, tmp_path, monkeypatch):
        attempted = []

        def fake_download(url, dest_path, **kwargs):
            attempted.append(url)
            if len(attempted) == 1:
                raise ConnectionError('primary unreachable')
            with open(dest_path, 'wb') as f:
                f.write(b'weights')

        monkeypatch.setattr(model_store, 'download_file', fake_download)
        monkeypatch.setitem(const.MODEL_REGISTRY, model, const.ModelInfo(url='https://gh.invalid/a.onnx', sha256=''))

        path = verify_model_weights(model, root=str(tmp_path))
        assert len(attempted) == 2
        assert attempted[1].startswith(const.HF_MIRROR_URL)
        assert os.path.exists(path)

    def test_no_fallback_when_primary_succeeds(self, model, tmp_path, monkeypatch):
        attempted = []

        def fake_download(url, dest_path, **kwargs):
            attempted.append(url)
            with open(dest_path, 'wb') as f:
                f.write(b'weights')

        monkeypatch.setattr(model_store, 'download_file', fake_download)
        monkeypatch.setitem(const.MODEL_REGISTRY, model, const.ModelInfo(url='https://gh.invalid/a.onnx', sha256=''))

        verify_model_weights(model, root=str(tmp_path))
        assert attempted == ['https://gh.invalid/a.onnx']

    def test_all_sources_failing_raises(self, model, tmp_path, monkeypatch):
        def fake_download(url, dest_path, **kwargs):
            raise ConnectionError('unreachable')

        monkeypatch.setattr(model_store, 'download_file', fake_download)
        monkeypatch.setitem(const.MODEL_REGISTRY, model, const.ModelInfo(url='https://gh.invalid/a.onnx', sha256=''))

        with pytest.raises(ConnectionError, match='GH Releases and HF Mirror'):
            verify_model_weights(model, root=str(tmp_path))

    def test_unknown_model_still_raises_value_error(self, tmp_path):
        class Bogus(str, Enum):
            X = 'bogus'

        with pytest.raises(ValueError, match='Unknown model identifier'):
            verify_model_weights(Bogus.X, root=str(tmp_path))
