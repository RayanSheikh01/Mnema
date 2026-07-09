from __future__ import annotations

from datetime import datetime
from typing import Any


def _yaml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_value(item) for item in value) + "]"
    text = str(value).replace('"', '\\"')
    return f"\"{text}\""


def render_note(frontmatter: dict[str, Any], body: str) -> str:
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, datetime):
            lines.append(f"{key}: \"{value.isoformat()}\"")
        else:
            lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    return "\n".join(lines)
