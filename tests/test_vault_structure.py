from __future__ import annotations

from pathlib import Path
import tempfile

from mnema_memory.config import AppConfig
from mnema_memory.service import MemoryService

# Fields Dataview queries rely on; every note must expose them in frontmatter.
REQUIRED_FRONTMATTER = {"type", "memory_id", "agent_id", "namespace", "timestamp", "tags", "importance"}


def build_service() -> MemoryService:
    root = Path(tempfile.mkdtemp())
    return MemoryService(
        AppConfig(
            vault_root=root / "vault",
            sqlite_path=root / "mnema.sqlite3",
            embedding_provider="local-hash",
        )
    )


def parse_frontmatter(note_path: Path) -> tuple[dict[str, str], str]:
    text = note_path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{note_path} missing frontmatter open"
    parts = text.split("---\n", 2)
    assert len(parts) == 3, f"{note_path} missing frontmatter close"
    frontmatter: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, parts[2]


def test_vault_notes_are_dataview_and_graph_ready() -> None:
    svc = build_service()
    a = svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "content": "Use JWT access tokens.",
            "tags": ["auth"],
        },
    )
    b = svc.router.call(
        "memory.remember",
        {
            "namespace": "org/proj/dev",
            "agent_id": "agent-x",
            "content": "Rotate refresh tokens.",
            "tags": ["auth"],
        },
    )
    svc.router.call(
        "memory.link",
        {"namespace": "org/proj/dev", "memory_id_a": a["memory_id"], "memory_id_b": b["memory_id"]},
    )
    summary = svc.router.call(
        "memory.summarize",
        {"namespace": "org/proj/dev", "agent_id": "agent-x", "topic": "auth"},
    )

    vault_root = svc.config.vault_root
    notes = list(vault_root.rglob("*.md"))
    assert len(notes) == 3  # two episodes + one summary

    # Every note carries the Dataview-queryable frontmatter with typed values.
    for note in notes:
        frontmatter, _ = parse_frontmatter(note)
        assert REQUIRED_FRONTMATTER.issubset(frontmatter.keys()), note
        assert frontmatter["type"].strip('"') in {"episode", "summary"}
        assert frontmatter["tags"].startswith("[") and frontmatter["tags"].endswith("]")
        float(frontmatter["importance"])  # importance must be a real number

    # Link produced reciprocal backlinks in both episodes' frontmatter.
    fa, _ = parse_frontmatter(Path(_path_of(svc, a["memory_id"])))
    fb, _ = parse_frontmatter(Path(_path_of(svc, b["memory_id"])))
    assert b["memory_id"] in fa["links"]
    assert a["memory_id"] in fb["links"]

    # Summary is a first-class note that links its sources via graph wikilinks.
    summary_fm, summary_body = parse_frontmatter(Path(_path_of(svc, summary["memory_id"])))
    assert summary_fm["type"].strip('"') == "summary"
    assert "derived_from" in summary_fm
    assert f"[[{a['memory_id']}]]" in summary_body
    assert f"[[{b['memory_id']}]]" in summary_body
    svc.close()


def _path_of(svc: MemoryService, memory_id: str) -> str:
    row = svc.conn.execute("SELECT path FROM memories WHERE id=?", (memory_id,)).fetchone()
    return row["path"]
