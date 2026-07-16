# Architecture

This Multi-Agent Turtlesim prototype is a **LangGraph hub-and-spoke*** system with a single orchestrator. The supervisor is the only agent that talks to the user and the only one that decides who works next; specialists never call each other.

*a routing pattern: one **hub** in the middle, and specialists as **spokes** that only talk to that hub, never to each other.

```
User ↔ supervisor ──(simple task / question: handled directly)──→ END
           │
           ├─→ planner_blueprint ─┐
           ├─→ planner_part ──────┤  every specialist returns
           ├─→ plan_reviewer ─────┤  to the supervisor
           └─→ navigator ─────────┘
```

## The five agents


| Agent               | LLM calls          | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `supervisor`        | 1 per user message | Analyses the request. Simple turtle tasks and questions it handles itself with motion tools. Drawing requests it hands to the pipeline via its `start_drawing` tool. On every revisit it routes **deterministically** (no LLM) through the fixed pipeline order.                                                                                                                                                                                          |
| `planner_blueprint` | 1 per mission      | Structured call, no tools: request → ordered snake_case parts + `side_length` budget.                                                                                                                                                                                                                                                                                                                                                                     |
| `planner_part`      | 1 per trial        | Structured call, no tools: movement steps (`set_pen` / `move_forward` / `rotate`) for the current part. Live pose, canvas bounds, pen-width guidance and progress are injected into the prompt deterministically. Trial 1 of a later part gets the last approved canvas image; replans receive reviewer feedback + the rejected trial image. The plan is then normalized (teleports stripped, negative moves fixed, pen width aligned with existing ink). |
| `plan_reviewer`     | ≤1 per trial       | Simulates the plan's ink, renders it on top of the ink already drawn, saves `captures/{shape}_{part}_trial{N}.png`. Empty plans and off-canvas ink are rejected for free (no LLM). Otherwise one vision call: approve, or reject with imperative fix instructions. If the vision call itself errors, the plan is approved without review.                                                                                                                 |
| `navigator`         | 0                  | Walks the approved plan on the real turtle via the ROS bridge and records ink on the mission's `PathTracker`.                                                                                                                                                                                                                                                                                                                                             |




## Data flow (drawing mission)

1. Supervisor's LLM turn calls `start_drawing` → routes to blueprint.
2. Blueprint writes `shape_name`, `parts`, `side_length` onto `MissionContext`.
3. Per part, the supervisor loops deterministically:
  - no plan → `planner_part`
  - plan, no verdict → `plan_reviewer`
  - rejected → feedback + trial image back to `planner_part` (trial N+1)
  - rejected `max_trials` times → skip part or draw anyway (`[review].fail_action`)
  - approved → `navigator`, then advance to the next part
4. All parts done → park the turtle, print the summary, END.

State lives in one mutable `MissionContext` (`utils/mission_state.py`) that every node reads and writes. Structured LLM outputs are Pydantic models (`pydantic_models.py`): `ShapeBlueprint`, `MovementPlanOutput`, `PlanReview`.

## Trial images = progressive story

`captures/` only ever contains `{shape}_{part}_trial{N}.png`. Each image shows the real ink so far plus the proposed part, so the sequence documents the build: which part failed review, how it was fixed, and how the drawing grew.

## Per-agent models (Not tested)

`llm.py` exposes `get_llm(agent)` where agent ∈ supervisor / blueprint / planner / reviewer. `config.toml` `[llm]` sets the default; optional`[llm.<agent>]` tables override provider and/or model per agent (e.g. a cheap model for the supervisor, a vision model for the reviewer). Providers: Gemini, OpenAI, Anthropic, Ollama via the official LangChain integration packages.

## Directory layout

```
src/
  main.py              CLI entry and interactive loop.
  app_settings.py      config.toml loader (+ env overrides).
  llm.py               Per-agent chat model factory (4 providers).
  pydantic_models.py   Structured output / artefact schemas.
  visualize_graph.py   Render the LangGraph as an image (pixi run graph).

  agents/
    supervisor.py        Orchestrator: ReAct turn + deterministic pipeline.
    planner_blueprint.py Ordered part list for a request.
    planner_part.py      Steps for one part (one call per trial).
    plan_reviewer.py     Trial PNG + vision verdict.
    navigator.py         Executes plan on the turtle (no LLM).

  tools/
    turtle_motion_tools.py  Pose / move / rotate / pen / clear / reset.

  workflow/
    graph.py           LangGraph wiring + run_mission().
    types.py           Graph state + routing labels.

  utils/
    agent_runtime.py   Bridge + mission notebook for @tools.
    mission_state.py   MissionContext (per-mission notebook).
    ros_bridge.py      ROS 2 turtlesim bridge (with mock backend).
    turtle_control.py  Low-level turtle commands.
    path_tracker.py    Ink segment memory.
    path_simulation.py Dry-run plans + canvas bounds check.
    drawing_render.py  Segments → PNG (Pillow).
    canvas_geometry.py 11×11 canvas constants.
    plan_execution.py  Deterministic plan runner.
    plan_normalize.py  Strip teleports, fix negative moves, align pen width.
    turtle_park.py     Park turtle after draw.
    cli_spinner.py     Terminal spinner while an LLM call runs.
```



## Configuration


| Source        | Purpose                                                                                |
| ------------- | -------------------------------------------------------------------------------------- |
| `config.toml` | Default + per-agent LLM, review loop (`enabled`, `max_trials`, `fail_action`), parking |
| `.env`        | API keys + optional LangSmith tracing (see[LANGSMITH.md](LANGSMITH.md))                |


Env overrides for one-off runs: `REVIEW_ENABLED`, `MAX_TRIALS`, `FAIL_ACTION`,
`PARK_TURTLE_AFTER_DRAW`, `APP_CONFIG` (alternate TOML path).

## Observability

Set the `LANGSMITH_*` variables in `.env` to get a full trace of every mission (each agent's LLM calls, prompts, images, tokens, latency) in LangSmith — no code changes needed. See [LANGSMITH.md](LANGSMITH.md).