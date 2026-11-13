# Multi-Agent Turtlesim

[中文版](README.zh.md)

This project contains a LangGraph multi-agent pipeline that draws shapes on ROS 2 turtlesim using configurable LLM/VLM providers (Gemini, OpenAI, Anthropic, or Ollama) with **one supervisor agent** orchestrating a small set of specialists agents.

## Setup

### Optional: Install pixi

Skip this if you already installed pixi for **session 1** — you do not need to reinstall. Jump ahead to `pixi install` below.

Otherwise install [pixi](https://pixi.sh) for your platform:

**macOS / Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows** (PowerShell)

```powershell
irm -useb https://pixi.sh/install.ps1 | iex
```

Then restart your shell (or `source ~/.bashrc` / `source ~/.zshrc`) so `pixi`
is on your `PATH`. Verify with:

```bash
pixi --version
```

### Install and configure

```bash
cd session_2
pixi install
cp .env.example .env          # API keys + optional LangSmith tracing (see docs/LANGSMITH.md)
```

## LLM providers

**Provider and model:** edit `[llm]` in `config.toml`.
**API key:** add the matching variable to `.env` (see `.env.example`).

```toml
[llm]
provider = "gemini"   # gemini | openai | anthropic | ollama
model = "gemini-3.1-flash-lite"
```

## Run

**Terminal 1** — simulator:

```bash
pixi run sim
```

**Terminal 2** — agents:

```bash
pixi run agents
```

Try:

- `Draw a blue square with side length 2.`
- `Draw a house`
- `Move forward 0.1`
- `Set pen to red`
- `How big is the canvas?`

## How it works

```
User ↔ SUPERVISOR (the only orchestrator, the only agent that talks to you)
          │  simple command / question → does it itself, done (1 LLM call)
          │  drawing request → runs the pipeline below, deterministically:
          │
          ├─→ planner_blueprint   parts + size            (1 LLM call/mission)
          ├─→ planner_part        steps for current part  (1 LLM call/trial)
          ├─→ plan_reviewer       trial PNG + vision check (≤1 LLM call/trial)
          └─→ navigator           draws on the turtle     (no LLM)
```

Every specialist reports back to the supervisor only — they never call each
other. After the first supervisor turn, routing through the pipeline is a
deterministic state machine, so **no tokens are spent on orchestration**.

### Progressive trial images

Before any ink hits the real canvas, the reviewer renders the plan **on top
of everything already drawn** and saves it as:

```
captures/{shape}_{part}_trial{N}.png     e.g. house_body_trial1.png
```

- Rejected trial → the image + fix instructions go back to the planner, and
the next attempt becomes `..._trial2.png`.
- Approved trial → the navigator draws it, and the next part's trial 1 image
builds on top of it.

These are the **only** images saved, so the captures folder reads as the
story of the drawing: what went wrong, how it was fixed, how the picture grew.

Free deterministic checks (empty plans, ink outside the 11×11 m canvas) are
rejected before the vision model is even called.

### Review settings (`config.toml`)

```toml
[review]
enabled = true        # REVIEW_ENABLED=0 to skip review for one run
max_trials = 5        # planning attempts per part (MAX_TRIALS)
fail_action = "skip"  # or "draw_anyway" when trials are exhausted (FAIL_ACTION)
```

After the mission the turtle parks in a clear corner
(`park.park_turtle_after_draw = true`).

## Project layout

```
src/
  main.py              CLI entry
  app_settings.py      config.toml loader (per-agent LLM config)
  llm.py               get_llm("supervisor"|"blueprint"|"planner"|"reviewer")
  pydantic_models.py   Structured LLM output schemas
  agents/              Agents (supervisor, planners, reviewer, navigator)
  workflow/            LangGraph wiring + run_mission()
  tools/               Turtle motion @tools for the supervisor
  utils/               ROS bridge, ink tracking, simulation, rendering, execution
docs/
  ARCHITECTURE.md      System design and data flow
  TOOLS.md             How tools work / how to add one
  LANGSMITH.md         Optional tracing: see every LLM call in a dashboard
```
