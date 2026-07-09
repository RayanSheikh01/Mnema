from __future__ import annotations

import os
from pathlib import Path
import tempfile

from mnema_memory.config import load_dotenv


def _tmp_env(body: str) -> Path:
    path = Path(tempfile.mkdtemp()) / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_key_value_pairs() -> None:
    path = _tmp_env('FOO_KEY=abc123\nBAR_KEY="quoted value"\n')
    os.environ.pop("FOO_KEY", None)
    os.environ.pop("BAR_KEY", None)
    load_dotenv(path)
    assert os.environ["FOO_KEY"] == "abc123"
    assert os.environ["BAR_KEY"] == "quoted value"
    os.environ.pop("FOO_KEY", None)
    os.environ.pop("BAR_KEY", None)


def test_existing_env_var_wins_over_file() -> None:
    path = _tmp_env("WINS_KEY=from-file\n")
    os.environ["WINS_KEY"] = "from-env"
    load_dotenv(path)
    assert os.environ["WINS_KEY"] == "from-env"
    os.environ.pop("WINS_KEY", None)


def test_skips_comments_and_blank_lines() -> None:
    path = _tmp_env("# a comment\n\n   \nGOOD_KEY=ok\n")
    os.environ.pop("GOOD_KEY", None)
    load_dotenv(path)
    assert os.environ["GOOD_KEY"] == "ok"
    assert "# a comment" not in os.environ
    os.environ.pop("GOOD_KEY", None)


def test_missing_file_is_noop() -> None:
    load_dotenv(Path(tempfile.mkdtemp()) / "does-not-exist.env")  # no raise
