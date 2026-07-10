"""Repair cp1252-mangled UTF-8 (mojibake) in vault markdown files.

On Windows the MCP server previously read stdin as cp1252, so UTF-8 bytes the
client sent (em-dash, curly quotes, …) were decoded wrong and persisted, e.g.
"—" landed on disk as "â€”". This reverses that: re-encode the mangled text as
cp1252 to recover the original UTF-8 bytes, then decode as UTF-8.

Usage:
    python scripts/fix_mojibake.py            # dry run, report only
    python scripts/fix_mojibake.py --apply    # rewrite files (backup to *.bak)
"""

from __future__ import annotations

import sys
from pathlib import Path

VAULT = Path(__file__).resolve().parent.parent / "vault"


def repair(text: str) -> str | None:
    """Return de-mangled text, or None if unchanged / not reversible."""
    try:
        fixed = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return fixed if fixed != text else None


def main() -> int:
    apply = "--apply" in sys.argv
    files = sorted(VAULT.rglob("*.md"))
    changed = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        fixed = repair(original)
        if fixed is None:
            continue
        changed += 1
        print(f"{'FIX ' if apply else 'WOULD FIX '}{path.relative_to(VAULT.parent)}")
        if apply:
            path.with_suffix(path.suffix + ".bak").write_text(original, encoding="utf-8")
            path.write_text(fixed, encoding="utf-8", newline="\n")
    print(f"\n{changed} file(s) {'fixed' if apply else 'need fixing'} of {len(files)} scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
