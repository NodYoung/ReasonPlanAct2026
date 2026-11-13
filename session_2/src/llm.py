"""
为每个 agent 构建 LangChain 聊天模型 —— Gemini / OpenAI / Anthropic / Ollama。

``get_llm("supervisor")`` 返回该 agent 的聊天模型，使用其在
``config.toml`` 中的 ``[llm.<agent>]`` 覆盖配置，若无则回退到 ``[llm]``。
API 密钥来自 ``.env``（参见 ``.env.example``）；Ollama 无需密钥。

采样参数保持 provider 的默认值 —— 当前的模型家族
（Gemini 3.x、GPT-5.x、Claude 5）推荐使用默认 temperature。
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
    # codex：路由器密钥来自环境变量 OPENAI_API_KEY 或 ~/.codex/auth.json
    "codex": ("OPENAI_API_KEY",),
}


def get_api_key(provider: LlmProvider) -> str | None:
    """从环境中返回 ``provider`` 的 API 密钥（如果有的话）。"""
    for env_name in _API_KEY_ENV[provider]:
        value = os.getenv(env_name)
        if value:
            return value
    if provider == "codex":
        # codex 密钥可能只存在于 ~/.codex/auth.json 中（不在环境变量里）
        from src.codex_backend import load_codex_settings

        try:
            key, _model, _base = load_codex_settings()
            return key
        except Exception:
            return None
    return None


def api_key_hint(provider: LlmProvider) -> str:
    """缺少凭据时的一行修复提示。"""
    if provider == "ollama":
        return f"Start Ollama locally ({get_config().llm.ollama_base_url})"
    if provider == "codex":
        return "Set OPENAI_API_KEY in .env/env or provide ~/.codex/auth.json"
    return f"Set {' or '.join(_API_KEY_ENV[provider])} in .env"


def missing_providers() -> list[LlmProvider]:
    """任意 agent 使用到的、但未配置 API 密钥的 provider。"""
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
    if provider == "codex":
        from src.codex_backend import build_codex_model

        return build_codex_model(model)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_llm(agent: str = "supervisor") -> BaseChatModel:
    """``agent``（supervisor / blueprint / planner / reviewer）的聊天模型。"""
    cfg: LlmConfig = get_config().llm_for(agent)
    return _build_model(cfg.provider, cfg.model, cfg.ollama_base_url)


def describe_models() -> str:
    """简短、便于人类阅读的摘要，说明每个 agent 使用哪个模型。"""
    cfg = get_config()
    lines = [f"default: {cfg.llm.provider}/{cfg.llm.model}"]
    for role in LLM_AGENT_ROLES:
        agent_cfg = cfg.llm_for(role)
        if agent_cfg != cfg.llm:
            lines.append(f"{role}: {agent_cfg.provider}/{agent_cfg.model}")
    return ", ".join(lines)
