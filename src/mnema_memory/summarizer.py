from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Any


LOGGER = logging.getLogger("mnema_memory")

# Source shape: a list of {"title": str, "content": str} dicts. Kept as plain
# dicts so the seam has no import coupling to the service layer.
Source = dict[str, str]

_SYSTEM_PROMPT = (
    "You summarize an agent's memory notes into a concise semantic summary. "
    "Extract the salient points, decisions made, unresolved items, and key "
    "entities or topics. Respond with GitHub-flavored markdown that begins with "
    "a '### Key Points' heading followed by short bullet points. Do not add a "
    "preamble, a title, or any '### Derived From' section — those are added by "
    "the caller."
)


def _render_sources(topic: str, sources: list[Source]) -> str:
    lines = [f"Topic: {topic}", "", "Source notes:"]
    for source in sources:
        lines.append(f"## {source['title']}")
        lines.append(source["content"].strip())
        lines.append("")
    return "\n".join(lines)


class SummaryGenerator(ABC):
    @abstractmethod
    def summarize(self, topic: str, sources: list[Source]) -> str:
        """Return the markdown body of a summary (starting at '### Key Points').

        The caller wraps this with the topic header and the deterministic
        'Derived From' wikilinks — the generator only writes prose."""
        raise NotImplementedError


class ExtractiveSummaryGenerator(SummaryGenerator):
    """Dependency-free default: one bullet per source, truncated excerpt.

    Byte-compatible with the pre-v5 summarize output so the offline default is
    unchanged."""

    def __init__(self, max_chars: int = 160) -> None:
        self.max_chars = max_chars

    def summarize(self, topic: str, sources: list[Source]) -> str:
        bullets = [
            f"- {source['title']}: {source['content'][: self.max_chars].strip()}"
            for source in sources
        ]
        return "\n".join(["### Key Points", *bullets])


class OpenAISummaryGenerator(SummaryGenerator):
    def __init__(self, model: str) -> None:
        self.model = model or "gpt-4o-mini"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai summary provider")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed; install with `pip install .[openai]`"
            ) from exc
        self._client: Any = OpenAI(api_key=api_key)

    def summarize(self, topic: str, sources: list[Source]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _render_sources(topic, sources)},
            ],
        )
        body = response.choices[0].message.content.strip()
        LOGGER.debug("openai summary generated model=%s sources=%s", self.model, len(sources))
        return body


class AnthropicSummaryGenerator(SummaryGenerator):
    def __init__(self, model: str) -> None:
        self.model = model or "claude-haiku-4-5"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic summary provider")
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package is not installed; install with `pip install .[anthropic]`"
            ) from exc
        self._client: Any = anthropic.Anthropic(api_key=api_key)

    def summarize(self, topic: str, sources: list[Source]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _render_sources(topic, sources)}],
        )
        body = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        LOGGER.debug("anthropic summary generated model=%s sources=%s", self.model, len(sources))
        return body


_EXTRACTIVE_ALIASES = {"extractive", "", "none", "local"}
_OPENAI_ALIASES = {"openai", "openai-api"}
_ANTHROPIC_ALIASES = {"anthropic", "claude"}


def build_summary_generator(provider: str, model: str = "") -> SummaryGenerator:
    normalized = provider.strip().lower()
    if normalized in _EXTRACTIVE_ALIASES:
        return ExtractiveSummaryGenerator()
    if normalized in _OPENAI_ALIASES:
        return OpenAISummaryGenerator(model)
    if normalized in _ANTHROPIC_ALIASES:
        return AnthropicSummaryGenerator(model)
    raise ValueError(f"unsupported summary provider: {provider}")
