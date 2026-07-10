from __future__ import annotations

import sys
import types

import pytest

from mnema_memory.summarizer import (
    AnthropicSummaryGenerator,
    ExtractiveSummaryGenerator,
    OpenAISummaryGenerator,
    build_summary_generator,
)


def test_extractive_output_shape() -> None:
    gen = ExtractiveSummaryGenerator(max_chars=10)
    body = gen.summarize(
        "auth",
        [
            {"title": "JWT", "content": "use jwt access tokens for everything and more"},
            {"title": "Refresh", "content": "rotate refresh tokens"},
        ],
    )
    lines = body.splitlines()
    assert lines[0] == "### Key Points"
    assert lines[1].startswith("- JWT: ")
    # Truncated to max_chars.
    assert "use jwt ac" in lines[1]
    assert lines[2] == "- Refresh: rotate ref"[: len("- Refresh: ") + 10]


@pytest.mark.parametrize(
    "provider,cls",
    [
        ("extractive", ExtractiveSummaryGenerator),
        ("", ExtractiveSummaryGenerator),
        ("none", ExtractiveSummaryGenerator),
    ],
)
def test_factory_extractive_aliases(provider: str, cls: type) -> None:
    assert isinstance(build_summary_generator(provider), cls)


def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported summary provider"):
        build_summary_generator("gemini")


# ---- fake SDK modules --------------------------------------------------------


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, capture: dict) -> None:
    module = types.ModuleType("openai")

    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str) -> None:
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, model, messages):
            capture["model"] = model
            capture["messages"] = messages
            return _Resp("### Key Points\n- synthesized openai point")

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class OpenAI:
        def __init__(self, api_key: str) -> None:
            self.chat = _Chat()

    module.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, capture: dict) -> None:
    module = types.ModuleType("anthropic")

    class _Block:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Resp:
        def __init__(self, text: str) -> None:
            self.content = [_Block(text)]

    class _Messages:
        def create(self, model, max_tokens, system, messages):
            capture["model"] = model
            capture["system"] = system
            capture["messages"] = messages
            return _Resp("### Key Points\n- synthesized claude point")

    class Anthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = _Messages()

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_openai_generator_sends_sources_and_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    capture: dict = {}
    _install_fake_openai(monkeypatch, capture)
    gen = build_summary_generator("openai", "gpt-4o-mini")
    assert isinstance(gen, OpenAISummaryGenerator)
    body = gen.summarize("auth", [{"title": "JWT", "content": "use jwt tokens"}])
    assert body == "### Key Points\n- synthesized openai point"
    assert capture["model"] == "gpt-4o-mini"
    # Source content reached the request.
    user_msg = capture["messages"][-1]["content"]
    assert "use jwt tokens" in user_msg
    assert "auth" in user_msg


def test_anthropic_generator_defaults_to_haiku(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    capture: dict = {}
    _install_fake_anthropic(monkeypatch, capture)
    gen = build_summary_generator("claude")  # no model -> default
    assert isinstance(gen, AnthropicSummaryGenerator)
    body = gen.summarize("auth", [{"title": "JWT", "content": "rotate refresh tokens"}])
    assert body == "### Key Points\n- synthesized claude point"
    assert capture["model"] == "claude-haiku-4-5"
    assert "rotate refresh tokens" in capture["messages"][-1]["content"]


def test_openai_generator_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_summary_generator("openai")


def test_anthropic_generator_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_summary_generator("anthropic")
