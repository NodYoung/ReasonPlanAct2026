"""
Application settings loaded from ``config.toml`` (+ select env overrides).

Behaviour knobs live in ``config.toml``; secrets stay in ``.env``.
``load_config()`` is called once at process start (``main.py``); later code
uses ``get_config()`` for a frozen snapshot.

``[llm]`` sets the default provider/model for every agent. Optional
``[llm.supervisor]`` / ``[llm.blueprint]`` / ``[llm.planner]`` /
``[llm.reviewer]`` tables override provider and/or model per agent.

``APP_CONFIG`` env var can point at an alternate TOML path (useful in tests).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

LlmProvider = Literal["gemini", "openai", "anthropic", "ollama"]
FailAction = Literal["skip", "draw_anyway"]

PROVIDERS = ("gemini", "openai", "anthropic", "ollama")
LLM_AGENT_ROLES = ("supervisor", "blueprint", "planner", "reviewer")

_CONFIG_PATH = Path(os.getenv("APP_CONFIG", "config.toml"))


@dataclass(frozen=True)
class LlmConfig:
    """Provider + model for one agent (or the default for all agents)."""

    provider: LlmProvider
    model: str
    ollama_base_url: str


@dataclass(frozen=True)
class ReviewConfig:
    """Pre-draw plan review loop (trial image + vision check)."""

    enabled: bool
    max_trials: int
    fail_action: FailAction


@dataclass(frozen=True)
class AppConfig:
    """Frozen bag of all sections from ``config.toml`` (+ env overlays)."""

    llm: LlmConfig
    llm_agents: dict[str, LlmConfig] = field(default_factory=dict)
    review: ReviewConfig = ReviewConfig(enabled=True, max_trials=3, fail_action="skip")
    park_after_draw: bool = True

    def llm_for(self, agent: str) -> LlmConfig:
        """LLM config for ``agent`` (supervisor/blueprint/planner/reviewer)."""
        return self.llm_agents.get(agent, self.llm)


_config: AppConfig | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _parse_provider(raw: object, where: str) -> LlmProvider:
    provider = str(raw or "").strip().lower()
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"Invalid or missing provider {provider!r} in {where} "
            f"(expected one of: {', '.join(PROVIDERS)})."
        )
    return provider  # type: ignore[return-value]


def _load_toml() -> dict:
    if not _CONFIG_PATH.is_file():
        return {}
    with _CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def load_config(*, reload: bool = False) -> AppConfig:
    """Load and cache ``AppConfig``. Pass ``reload=True`` to re-read the file."""
    global _config
    if _config is not None and not reload:
        return _config

    load_dotenv()
    data = _load_toml()
    llm = data.get("llm", {})

    provider = _parse_provider(llm.get("provider"), "[llm]")
    model = str(llm.get("model", "")).strip()
    if not model:
        raise RuntimeError(f"Missing required [llm].model in config.toml for provider {provider!r}.")
    ollama_url = str(llm.get("ollama_base_url", "http://localhost:11434"))
    default = LlmConfig(provider=provider, model=model, ollama_base_url=ollama_url)

    # Per-agent overrides: [llm.supervisor], [llm.blueprint], ... fall back to [llm].
    agents: dict[str, LlmConfig] = {}
    for role in LLM_AGENT_ROLES:
        section = llm.get(role)
        if not isinstance(section, dict):
            continue
        agents[role] = LlmConfig(
            provider=(
                _parse_provider(section["provider"], f"[llm.{role}]")
                if "provider" in section
                else default.provider
            ),
            model=str(section.get("model", default.model)).strip() or default.model,
            ollama_base_url=str(section.get("ollama_base_url", ollama_url)),
        )

    review = data.get("review", {})
    fail_action = os.getenv("FAIL_ACTION", review.get("fail_action", "skip"))
    if fail_action not in ("skip", "draw_anyway"):
        raise RuntimeError(f"Invalid [review].fail_action {fail_action!r} (skip | draw_anyway).")

    _config = AppConfig(
        llm=default,
        llm_agents=agents,
        review=ReviewConfig(
            enabled=_env_bool("REVIEW_ENABLED", review.get("enabled", True)),
            max_trials=max(1, _env_int("MAX_TRIALS", review.get("max_trials", 3))),
            fail_action=fail_action,  # type: ignore[arg-type]
        ),
        park_after_draw=_env_bool(
            "PARK_TURTLE_AFTER_DRAW", data.get("park", {}).get("park_turtle_after_draw", True)
        ),
    )
    return _config


def get_config() -> AppConfig:
    """Return the cached config, loading it on first use if needed."""
    return _config if _config is not None else load_config()
