"""
Build LangChain chat models per agent — Gemini / OpenAI / Anthropic / Ollama.

``get_llm("supervisor")`` returns a chat model for that agent using its
``[llm.<agent>]`` override from ``config.toml``, falling back to ``[llm]``.
API keys come from ``.env`` (see ``.env.example``); Ollama needs no key.

Sampling parameters are left at provider defaults — current model families
(Gemini 3.x, GPT-5.x, Claude 5) recommend default temperature.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.app_settings import LLM_AGENT_ROLES, LlmConfig, LlmProvider, get_config

_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
}


def get_api_key(provider: LlmProvider) -> str | None:
    """Return the API key for ``provider`` from the environment, if any."""
    for env_name in _API_KEY_ENV[provider]:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def api_key_hint(provider: LlmProvider) -> str:
    """One-line fix hint for missing credentials."""
    if provider == "ollama":
        return f"Start Ollama locally ({get_config().llm.ollama_base_url})"
    return f"Set {' or '.join(_API_KEY_ENV[provider])} in .env"


def missing_providers() -> list[LlmProvider]:
    """Providers used by any agent that have no API key configured."""
    cfg = get_config()
    used = {cfg.llm.provider} | {cfg.llm_for(role).provider for role in LLM_AGENT_ROLES}
    return [p for p in sorted(used) if p != "ollama" and not get_api_key(p)]


def _require_api_key(provider: LlmProvider) -> str:
    key = get_api_key(provider)
    if not key:
        raise RuntimeError(f"{provider.title()} API key not found. {api_key_hint(provider)}")
    return key


@lru_cache(maxsize=8)
def _build_model(provider: LlmProvider, model: str, ollama_base_url: str) -> BaseChatModel:
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, api_key=_require_api_key("gemini"))
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=_require_api_key("openai"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=_require_api_key("anthropic"))
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=ollama_base_url)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_llm(agent: str = "supervisor") -> BaseChatModel:
    """Chat model for ``agent`` (supervisor / blueprint / planner / reviewer)."""
    cfg: LlmConfig = get_config().llm_for(agent)
    return _build_model(cfg.provider, cfg.model, cfg.ollama_base_url)


def describe_models() -> str:
    """Short human-readable summary of which model each agent uses."""
    cfg = get_config()
    lines = [f"default: {cfg.llm.provider}/{cfg.llm.model}"]
    for role in LLM_AGENT_ROLES:
        agent_cfg = cfg.llm_for(role)
        if agent_cfg != cfg.llm:
            lines.append(f"{role}: {agent_cfg.provider}/{agent_cfg.model}")
    return ", ".join(lines)
