from __future__ import annotations

from pathlib import Path

import pytest

from mnema_memory.config import AppConfig


def test_loads_local_embedding_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "mnema.toml"
    config_path.write_text(
        """
[mnema]
vault_root = "vault"
sqlite_path = "mnema.sqlite3"
embedding_provider = "local"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
local_model_cache = "models"
local_files_only = true
local_device = "cpu"
local_batch_size = 8
""".strip(),
        encoding="utf-8",
    )
    for key in (
        "MNEMA_EMBEDDING_PROVIDER",
        "MNEMA_EMBEDDING_MODEL",
        "MNEMA_LOCAL_MODEL_CACHE",
        "MNEMA_LOCAL_FILES_ONLY",
        "MNEMA_LOCAL_DEVICE",
        "MNEMA_LOCAL_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    # .env is loaded by AppConfig; explicit environment values must win in this test.
    monkeypatch.setenv("MNEMA_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("MNEMA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    config = AppConfig.load(config_path)

    assert config.embedding_provider == "local"
    assert config.local_model_cache == Path("models").resolve()
    assert config.local_files_only is True
    assert config.local_device == "cpu"
    assert config.local_batch_size == 8


def test_local_batch_size_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMA_LOCAL_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="local_batch_size"):
        AppConfig.load()
