"""
llm_config.py — LLM backend abstraction
========================================
Switch backends with a single environment variable:

  export LLM_PROVIDER=codex       # Codex Responses router (default — reads ~/.codex)
  export LLM_PROVIDER=google      # Gemini Flash (free via AI Studio)
  export LLM_PROVIDER=ollama      # Local model, no API key needed
  export LLM_PROVIDER=openai      # GPT-4o-mini
  export LLM_PROVIDER=anthropic   # Claude Haiku

Copy .env.example to .env and fill in your key, or export variables directly.

The default `codex` provider talks to the same Codex-compatible Responses
backend used elsewhere in this repo. It automatically reads the local Codex
config files:

  ~/.codex/auth.json    -> API key   (OPENAI_API_KEY)
  ~/.codex/config.toml  -> model / model_provider / base_url

Environment variables take precedence over the config files.
"""

import ast
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()


def _load_toml(path: Path) -> dict:
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
            section[key] = value.strip("\"'")
    return result


def _load_codex_settings() -> tuple[str, str, str | None]:
    """Resolve (api_key, model, base_url) from env vars and ~/.codex config."""
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

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it explicitly or provide a Codex "
            f"auth file at {auth_path}."
        )
    return api_key, model, base_url


def _codex_headers() -> tuple[dict, dict]:
    """Build the Codex-specific headers/extra_body the router requires."""
    request_id = str(uuid4())
    session_id = os.environ.get("CODEX_SESSION_ID", request_id)
    installation_id = os.environ.get(
        "CODEX_INSTALLATION_ID", "codex-router-example"
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


def get_llm(temperature: float = 0.7):
    """
    Return a LangChain chat model for the configured provider.

    All returned objects share the same interface:
      .invoke(messages)        -> AIMessage
      .bind(**kwargs)          -> new model with extra params (max_tokens, stop, …)
      .stream(messages)        -> iterator of AIMessageChunk

    Parameters
    ----------
    temperature : float
        0.0 = greedy / deterministic, 1.0 = sample from the distribution.
        Use low values (0.0–0.2) for structured agent outputs (tool calls, JSON).
        Use moderate values (0.5–0.8) for natural language reasoning steps.
    """
    provider = (os.getenv("LLM_PROVIDER") or "codex").lower().strip()

    if provider == "codex":
        from langchain_openai import ChatOpenAI

        api_key, model, base_url = _load_codex_settings()
        headers, extra_body = _codex_headers()
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            use_responses_api=True,
            default_headers=headers,
            extra_body=extra_body,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemma-4-31b-it"),  # gemini-3.1-flash-lite, gemma-4-31b-it
            temperature=temperature,
        )

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL")
        model = os.getenv("OLLAMA_MODEL")

        from ollama import Client
        from tqdm.asyncio import tqdm

        client = Client(host=base_url)
        with tqdm(
            desc=f'Ollama pulling {model}',
            unit='B', unit_scale=True, unit_divisor=1024,
            leave=False,
        ) as bar:
            for info in client.pull(model, stream=True):
                try:
                    bar.total = info.total
                    bar.set_postfix_str(info.status)
                    bar.update((info.completed or 0) - bar.n)
                except AttributeError:
                    pass

        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'.\n"
            f"Choose from: codex | google | ollama | openai | anthropic\n"
            f"Set it with:  export LLM_PROVIDER=codex"
        )


def check_connection() -> bool:
    """
    Quick smoke-test: send a single token request to verify credentials work.
    """
    try:
        llm = get_llm(temperature=0.7)
        ping = "ping"
        pong = llm.invoke(ping).text
        print(pong)
        return True
    except Exception as e:
        print(f"\n[llm_config] Connection failed for provider '{os.getenv('LLM_PROVIDER')}'.")
        print(f"  Error: {e}")
        print(f"  Check your API key or that Ollama is running.\n")
        return False

if __name__ == "__main__":
    check_connection()
