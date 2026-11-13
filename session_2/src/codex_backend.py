"""
session_2 的 Codex Responses 后端。

构建一个 LangChain ``ChatOpenAI``，它与 Codex 兼容的 Responses
路由器通信（本仓库其他地方使用的也是同一个后端），从本地的
Codex 配置文件中读取 key / model / base_url：

    ~/.codex/auth.json    -> API 密钥（OPENAI_API_KEY）
    ~/.codex/config.toml  -> model / model_provider / base_url

该路由器需要 Codex 专用的请求头（User-Agent: codex_exec/...、originator、
x-codex-turn-metadata、……）以及 ``store=false``；缺少它们会返回 HTTP 403。
环境变量优先于配置文件。

返回的模型使用 Responses API 并支持图像输入（视觉），
这是 reviewer agent 所需要的。
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from uuid import uuid4


def _load_toml(path: Path) -> dict:
    """读取 Codex 配置文件所使用的 TOML 的一小部分子集。"""
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
            section[key] = value.strip("\"'")
    return result


def load_codex_settings() -> tuple[str, str, str | None]:
    """从环境变量和 ~/.codex 配置中解析出 (api_key, model, base_url)。"""
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
        config = _load_toml(config_path)

    provider_name = os.environ.get(
        "OPENAI_MODEL_PROVIDER", config.get("model_provider", "")
    )
    provider = config.get("model_providers", {}).get(provider_name, {})

    model = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("ROBOT_AGENT_LLM_MODEL")
        or config.get("model")
        or "gpt-6-astra"
    )
    base_url = os.environ.get("OPENAI_BASE_URL") or provider.get("base_url")
    return api_key, model, base_url


def _codex_headers() -> tuple[dict, dict]:
    """构建路由器所需的 Codex 专用请求头 / extra_body。"""
    request_id = str(uuid4())
    session_id = os.environ.get("CODEX_SESSION_ID", request_id)
    installation_id = os.environ.get(
        "CODEX_INSTALLATION_ID", "codex-router-session2"
    )
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
    headers = {
        "Accept": "text/event-stream",
        "User-Agent": "codex_exec/0.146.0",
        "originator": "codex_exec",
        "x-client-request-id": request_id,
        "x-codex-window-id": f"{request_id}:0",
        "session-id": session_id,
        "thread-id": session_id,
        "x-codex-turn-metadata": turn_metadata,
    }
    extra_body = {
        "store": False,
        "client_metadata": {
            "x-codex-installation-id": installation_id,
            "x-codex-window-id": f"{request_id}:0",
            "session_id": session_id,
            "thread_id": session_id,
            "x-codex-turn-metadata": turn_metadata,
        },
    }
    return headers, extra_body


def _strip_surrogates(text: str) -> str:
    """移除字符串里孤立的 UTF-16 代理项（0xD800–0xDFFF）。

    这类字符无法编码为 UTF-8，若进入消息历史会在后续 json 序列化时抛
    UnicodeEncodeError: surrogates not allowed。
    """
    if not text:
        return text
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def _sanitize_message_content(content):
    """清洗 AIMessage.content（可能是 str 或 分段 list）里的孤立代理项。"""
    if isinstance(content, str):
        return _strip_surrogates(content)
    if isinstance(content, list):
        cleaned = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part = {**part, "text": _strip_surrogates(part["text"])}
            cleaned.append(part)
        return cleaned
    return content


def build_codex_model(model: str):
    """返回一个为 Codex Responses 路由器配置好的 ChatOpenAI。

    ``model`` 来自 config.toml；如果为空，则回退到 ``load_codex_settings``
    内部基于 Codex 配置文件 / 环境变量的解析结果。
    """
    from langchain_openai import ChatOpenAI

    class _SurrogateSafeChatOpenAI(ChatOpenAI):
        """在模型返回进入消息历史之前，清洗孤立代理项字符。

        防御 Codex SSE 流式分片把 emoji/多字节字符切断后留下的孤立 surrogate，
        避免它们污染历史并在后续序列化时触发 UnicodeEncodeError。
        """

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for gen in result.generations:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    msg.content = _sanitize_message_content(msg.content)
            return result

    api_key, resolved_model, base_url = load_codex_settings()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY for codex provider. Set it in .env/env or "
            "provide ~/.codex/auth.json."
        )
    chosen_model = model or resolved_model
    headers, extra_body = _codex_headers()
    return _SurrogateSafeChatOpenAI(
        model=chosen_model,
        api_key=api_key,
        base_url=base_url,
        use_responses_api=True,
        default_headers=headers,
        extra_body=extra_body,
        # max_retries=0：Codex router 以 SSE 流式返回，输出里的 emoji/多字节字符
        # 若在分片边界被切断，拼接后可能留下孤立的 UTF-16 代理项（surrogate）。
        # openai SDK 在「重试」时会把包含该字符的消息历史重新 json 序列化，触发
        # UnicodeEncodeError: surrogates not allowed。关闭自动重试即可避开这条崩溃路径。
        # （_SurrogateSafeChatOpenAI 还会在返回进入历史前清洗这类字符，双重防御。）
        max_retries=0,
    )
