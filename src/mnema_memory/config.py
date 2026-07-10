from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib
from typing import Any


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Zero-dependency. Already-set environment variables win over the file, so an
    exported OPENAI_API_KEY still overrides the .env value. Lines that are blank
    or start with '#' are skipped; surrounding quotes on the value are stripped.
    """
    path = dotenv_path or (Path.cwd() / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AppConfig:
    vault_root: Path
    sqlite_path: Path
    default_namespace: str = "default/project/dev"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    summary_provider: str = "extractive"
    summary_model: str = ""
    local_model_cache: Path | None = None
    local_files_only: bool = False
    local_device: str | None = None
    local_batch_size: int = 32
    vector_backend: str = "numpy"
    dedup_enabled: bool = True
    dedup_threshold: float = 0.95
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef: int = 64

    @staticmethod
    def load(config_path: Path | None = None) -> "AppConfig":
        load_dotenv()
        file_data: dict[str, Any] = {}
        if config_path is not None:
            with config_path.open("rb") as handle:
                file_data = tomllib.load(handle).get("mnema", {})

        vault_root = Path(
            os.environ.get("MNEMA_VAULT_ROOT", file_data.get("vault_root", "./vault"))
        ).resolve()
        sqlite_path = Path(
            os.environ.get("MNEMA_SQLITE_PATH", file_data.get("sqlite_path", "./mnema.db"))
        ).resolve()
        default_namespace = os.environ.get(
            "MNEMA_DEFAULT_NAMESPACE", file_data.get("default_namespace", "default/project/dev")
        )
        from .embeddings import canonical_provider_name

        embedding_provider = canonical_provider_name(
            os.environ.get(
                "MNEMA_EMBEDDING_PROVIDER", file_data.get("embedding_provider", "openai")
            )
        )
        embedding_model = os.environ.get(
            "MNEMA_EMBEDDING_MODEL", file_data.get("embedding_model", "text-embedding-3-small")
        )
        summary_provider = os.environ.get(
            "MNEMA_SUMMARY_PROVIDER", file_data.get("summary_provider", "extractive")
        )
        summary_model = os.environ.get(
            "MNEMA_SUMMARY_MODEL", file_data.get("summary_model", "")
        )
        local_model_cache_raw = os.environ.get(
            "MNEMA_LOCAL_MODEL_CACHE", file_data.get("local_model_cache")
        )
        local_model_cache = (
            Path(str(local_model_cache_raw)).expanduser().resolve()
            if local_model_cache_raw and str(local_model_cache_raw).strip()
            else None
        )
        local_device_raw = os.environ.get("MNEMA_LOCAL_DEVICE", file_data.get("local_device"))
        local_device = str(local_device_raw).strip() if local_device_raw else None
        vector_backend = os.environ.get(
            "MNEMA_VECTOR_BACKEND", file_data.get("vector_backend", "numpy")
        )

        def _bool(value: object, default: bool) -> bool:
            if value is None:
                return default
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        dedup_enabled = _bool(
            os.environ.get("MNEMA_DEDUP_ENABLED", file_data.get("dedup_enabled")),
            True,
        )
        dedup_threshold = float(
            os.environ.get("MNEMA_DEDUP_THRESHOLD", file_data.get("dedup_threshold", 0.95))
        )
        local_files_only = _bool(
            os.environ.get("MNEMA_LOCAL_FILES_ONLY", file_data.get("local_files_only")),
            False,
        )
        local_batch_size = int(
            os.environ.get("MNEMA_LOCAL_BATCH_SIZE", file_data.get("local_batch_size", 32))
        )
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        hnsw_m = int(os.environ.get("MNEMA_HNSW_M", file_data.get("hnsw_m", 16)))
        hnsw_ef_construction = int(
            os.environ.get("MNEMA_HNSW_EF_CONSTRUCTION", file_data.get("hnsw_ef_construction", 200))
        )
        hnsw_ef = int(os.environ.get("MNEMA_HNSW_EF", file_data.get("hnsw_ef", 64)))

        return AppConfig(
            vault_root=vault_root,
            sqlite_path=sqlite_path,
            default_namespace=default_namespace,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            summary_provider=summary_provider,
            summary_model=summary_model,
            local_model_cache=local_model_cache,
            local_files_only=local_files_only,
            local_device=local_device,
            local_batch_size=local_batch_size,
            vector_backend=vector_backend,
            dedup_enabled=dedup_enabled,
            dedup_threshold=dedup_threshold,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef=hnsw_ef,
        )
