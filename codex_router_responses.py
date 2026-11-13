#!/usr/bin/env python3
"""通过与项目相同的 Codex 兼容 Responses 后端调用 GPT 模型。

使用方法
--------

    # 先进入容器，再在 Conda base 环境中使用本地 Codex 配置。
    enter_ultron    # 进入容器 ultron_1013
    cd /home/robot/github/llm-misc/ReasonPlanAct2026
    eval "$(/data/miniforge3/bin/conda shell.bash hook)"
    conda activate base
    python codex_router_responses.py "你好，请介绍一下自己"

    # 也可以显式设置 API 密钥、Router 地址和模型。
    export OPENAI_API_KEY="你的 key"
    export OPENAI_BASE_URL="你的 Codex API Router/v1 地址"
    export OPENAI_MODEL="gpt-6-astra"
    python codex_router_responses.py "你好"

脚本会自动读取以下本地 Codex 配置文件：

    ~/.codex/auth.json    -> API 密钥（OPENAI_API_KEY）
    ~/.codex/config.toml  -> 模型（model）、服务提供方（model_provider）及地址（base_url）

环境变量的优先级高于配置文件中的值。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from uuid import uuid4


def load_toml(path: Path) -> dict:
    """Read the small subset of TOML used by Codex config files."""
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None

    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)

    result: dict = {"model_providers": {}}
    section: dict = result
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            parts = line[1:-1].split(".")
            section = result
            for part in parts:
                if part == "model_providers":
                    section = result.setdefault(part, {})
                else:
                    section = section.setdefault(part, {})
            continue
        if "=" not in line:
            continue
        key, value = (item.strip() for item in line.split("=", 1))
        try:
            section[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            section[key] = value.strip('"\'')
    return result


def load_settings() -> tuple[str, str, str | None]:
    auth_path = Path(
        os.environ.get("CODEX_AUTH_FILE", "~/.codex/auth.json")
    ).expanduser()
    config_path = Path(
        os.environ.get("CODEX_CONFIG_FILE", "~/.codex/config.toml")
    ).expanduser()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and auth_path.is_file():
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        api_key = auth.get("OPENAI_API_KEY")

    config: dict = {}
    if config_path.is_file():
        config = load_toml(config_path)

    provider_name = os.environ.get(
        "OPENAI_MODEL_PROVIDER", config.get("model_provider", "")
    )
    providers = config.get("model_providers", {})
    provider = providers.get(provider_name, {})

    model = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("ROBOT_AGENT_LLM_MODEL")
        or config.get("model")
        or "gpt-6-astra"
    )
    base_url = os.environ.get("OPENAI_BASE_URL") or provider.get("base_url")

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it explicitly or provide a Codex "
            f"auth file at {auth_path}."
        )
    if not model:
        raise RuntimeError(
            "Missing model. Set OPENAI_MODEL or configure model in "
            f"{config_path}."
        )
    return api_key, model, base_url


def ask_codex(prompt: str) -> str:
    from openai import OpenAI

    api_key, model, base_url = load_settings()
    request_id = str(uuid4())
    session_id = os.environ.get("CODEX_SESSION_ID", request_id)
    installation_id = os.environ.get(
        "CODEX_INSTALLATION_ID", "codex-router-example"
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    request = {
        "model": model,
        "instructions": "Answer clearly and concisely.",
        "input": prompt,
        "max_output_tokens": int(
            os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "512")
        ),
        "store": False,
    }
    if base_url:
        turn_metadata = json.dumps(
            {
                "installation_id": installation_id,
                "session_id": session_id,
                "thread_id": session_id,
                "turn_id": request_id,
                "window_id": f"{request_id}:0",
                "request_kind": "turn",
                "thread_source": "user",
                "sandbox": "none",
            },
            separators=(",", ":"),
        )
        request.update(
            extra_headers={
                "Accept": "text/event-stream",
                "User-Agent": "codex_exec/0.146.0",
                "originator": "codex_exec",
                "x-client-request-id": request_id,
                "x-codex-window-id": f"{request_id}:0",
                "session-id": session_id,
                "thread-id": session_id,
                "x-codex-turn-metadata": turn_metadata,
            },
            extra_body={
                "client_metadata": {
                    "x-codex-installation-id": installation_id,
                    "x-codex-window-id": f"{request_id}:0",
                    "session_id": session_id,
                    "thread_id": session_id,
                    "x-codex-turn-metadata": turn_metadata,
                }
            },
        )
    response = client.responses.create(**request)
    print(f"model: {model}")
    print(f"response_id: {response.id}")
    return response.output_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call a GPT model through the Codex Responses backend."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text; defaults to a short Chinese greeting.",
    )
    args = parser.parse_args()
    prompt = " ".join(args.prompt).strip() or "用一句话介绍你自己。"

    try:
        print(ask_codex(prompt))
    except Exception as error:
        print(f"request failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
