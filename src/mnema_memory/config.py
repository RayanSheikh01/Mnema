from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


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
    vector_backend: str = "inmemory"

    @staticmethod
    def load(config_path: Path | None = None) -> "AppConfig":
        load_dotenv()
        file_data: dict[str, str] = {}
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
        embedding_provider = os.environ.get(
            "MNEMA_EMBEDDING_PROVIDER", file_data.get("embedding_provider", "openai")
        )
        embedding_model = os.environ.get(
            "MNEMA_EMBEDDING_MODEL", file_data.get("embedding_model", "text-embedding-3-small")
        )
        vector_backend = os.environ.get(
            "MNEMA_VECTOR_BACKEND", file_data.get("vector_backend", "inmemory")
        )

        return AppConfig(
            vault_root=vault_root,
            sqlite_path=sqlite_path,
            default_namespace=default_namespace,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_backend=vector_backend,
        )
