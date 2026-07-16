"""
llm_config.py — LLM backend abstraction
========================================
Switch backends with a single environment variable:

  export LLM_PROVIDER=google      # Gemini Flash (default — free via AI Studio)
  export LLM_PROVIDER=ollama      # Local model, no API key needed
  export LLM_PROVIDER=openai      # GPT-4o-mini
  export LLM_PROVIDER=anthropic   # Claude Haiku

Copy .env.example to .env and fill in your key, or export variables directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()


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
    provider = os.getenv("LLM_PROVIDER").lower().strip()

    if provider == "google":
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
            f"Choose from: google | ollama | openai | anthropic\n"
            f"Set it with:  export LLM_PROVIDER=google"
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
