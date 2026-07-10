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
    # Recall ranking (defaults reproduce the pre-v6 hardcoded blend).
    rank_weight_vector: float = 0.6
    rank_weight_recency: float = 0.2
    rank_weight_importance: float = 0.1
    rank_weight_tag: float = 0.1
    recency_half_life_days: float = 1.0
    # Write-time importance heuristic (opt-in; explicit caller value always wins).
    auto_importance: bool = False
    # Retention sweep (opt-in; reversible forget, never hard-delete).
    retention_enabled: bool = False
    retention_max_age_days: float = 365.0
    retention_min_importance: float = 0.25
    retention_exempt_summaries: bool = True

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

        def _float(env_key: str, toml_key: str, default: float) -> float:
            return float(os.environ.get(env_key, file_data.get(toml_key, default)))

        rank_weight_vector = _float("MNEMA_RANK_WEIGHT_VECTOR", "rank_weight_vector", 0.6)
        rank_weight_recency = _float("MNEMA_RANK_WEIGHT_RECENCY", "rank_weight_recency", 0.2)
        rank_weight_importance = _float(
            "MNEMA_RANK_WEIGHT_IMPORTANCE", "rank_weight_importance", 0.1
        )
        rank_weight_tag = _float("MNEMA_RANK_WEIGHT_TAG", "rank_weight_tag", 0.1)
        for name, weight in (
            ("rank_weight_vector", rank_weight_vector),
            ("rank_weight_recency", rank_weight_recency),
            ("rank_weight_importance", rank_weight_importance),
            ("rank_weight_tag", rank_weight_tag),
        ):
            if weight < 0:
                raise ValueError(f"{name} must be non-negative")
        recency_half_life_days = _float(
            "MNEMA_RECENCY_HALF_LIFE_DAYS", "recency_half_life_days", 1.0
        )
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        auto_importance = _bool(
            os.environ.get("MNEMA_AUTO_IMPORTANCE", file_data.get("auto_importance")),
            False,
        )
        retention_enabled = _bool(
            os.environ.get("MNEMA_RETENTION_ENABLED", file_data.get("retention_enabled")),
            False,
        )
        retention_max_age_days = _float(
            "MNEMA_RETENTION_MAX_AGE_DAYS", "retention_max_age_days", 365.0
        )
        retention_min_importance = _float(
            "MNEMA_RETENTION_MIN_IMPORTANCE", "retention_min_importance", 0.25
        )
        retention_exempt_summaries = _bool(
            os.environ.get(
                "MNEMA_RETENTION_EXEMPT_SUMMARIES",
                file_data.get("retention_exempt_summaries"),
            ),
            True,
        )

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
            rank_weight_vector=rank_weight_vector,
            rank_weight_recency=rank_weight_recency,
            rank_weight_importance=rank_weight_importance,
            rank_weight_tag=rank_weight_tag,
            recency_half_life_days=recency_half_life_days,
            auto_importance=auto_importance,
            retention_enabled=retention_enabled,
            retention_max_age_days=retention_max_age_days,
            retention_min_importance=retention_min_importance,
            retention_exempt_summaries=retention_exempt_summaries,
        )
