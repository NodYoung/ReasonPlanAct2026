"""
从 ``config.toml`` 加载应用设置（外加部分环境变量覆盖）。

行为相关的开关放在 ``config.toml`` 中；密钥保存在 ``.env`` 中。
``load_config()`` 在进程启动时调用一次（``main.py``）；后续代码
使用 ``get_config()`` 获取一份冻结的快照。

``[llm]`` 为每个 agent 设置默认的 provider/model。可选的
``[llm.supervisor]`` / ``[llm.blueprint]`` / ``[llm.planner]`` /
``[llm.reviewer]`` 表可按 agent 覆盖 provider 和/或 model。

``APP_CONFIG`` 环境变量可以指向另一个 TOML 路径（在测试中很有用）。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

LlmProvider = Literal["gemini", "openai", "anthropic", "ollama", "codex"]
FailAction = Literal["skip", "draw_anyway"]

PROVIDERS = ("gemini", "openai", "anthropic", "ollama", "codex")
LLM_AGENT_ROLES = ("supervisor", "blueprint", "planner", "reviewer")

_CONFIG_PATH = Path(os.getenv("APP_CONFIG", "config.toml"))


@dataclass(frozen=True)
class LlmConfig:
    """某个 agent 的 provider + model（或所有 agent 的默认值）。"""

    provider: LlmProvider
    model: str
    ollama_base_url: str


@dataclass(frozen=True)
class ReviewConfig:
    """绘制前的计划评审循环（试画图像 + 视觉检查）。"""

    enabled: bool
    max_trials: int
    fail_action: FailAction


@dataclass(frozen=True)
class AppConfig:
    """来自 ``config.toml`` 的所有配置段的冻结集合（外加环境变量叠加）。"""

    llm: LlmConfig
    llm_agents: dict[str, LlmConfig] = field(default_factory=dict)
    review: ReviewConfig = ReviewConfig(enabled=True, max_trials=3, fail_action="skip")
    park_after_draw: bool = True

    def llm_for(self, agent: str) -> LlmConfig:
        """返回 ``agent`` 的 LLM 配置（supervisor/blueprint/planner/reviewer）。"""
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
    """加载并缓存 ``AppConfig``。传入 ``reload=True`` 可重新读取文件。"""
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

    # 按 agent 覆盖：[llm.supervisor]、[llm.blueprint]、…… 回退到 [llm]。
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
    """返回缓存的配置，必要时在首次使用时加载它。"""
    return _config if _config is not None else load_config()
