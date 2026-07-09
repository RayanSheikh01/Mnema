from __future__ import annotations

from datetime import datetime, timezone
import re
import uuid


def generate_memory_id(now: datetime | None = None) -> str:
    current = now or datetime.now(tz=timezone.utc)
    return f"{current.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:12]}"


def slugify(value: str, max_length: int = 48) -> str:
    lowered = value.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = slug.strip("-")
    return slug[:max_length] or "memory"
