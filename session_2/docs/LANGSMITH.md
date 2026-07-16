# LangSmith — see what your agents are doing

## What is it?

When you type "Draw a house", several LLM calls happen behind the scenes: the supervisor analyses your request, the blueprint planner splits the house into parts, the part planner writes movement steps, the reviewer looks at trial images. All of that is invisible in the terminal.

[LangSmith](https://smith.langchain.com) is a free web dashboard that records every one of those calls — the exact prompt sent to the model, the images attached, the response, how long it took, and how many tokens it cost. Think of it as a flight recorder for the agents. Nothing about the app changes; it just reports what happened.

## Setup (5 minutes, no code changes)

1. **Create a free account** at
  [smith.langchain.com](https://smith.langchain.com) (sign up with Google or GitHub works fine). The free tier is more than enough for this project.
2. **Create an API key**: in LangSmith click **Settings (gear icon) → API Keys → Create API Key**. Copy the key — it starts with `lsv2_`.
3. **Put it in** `.env` (the same file that holds your LLM key):

```bash
 LANGSMITH_TRACING=true
 LANGSMITH_API_KEY=lsv2_...your-key-here...
 LANGSMITH_PROJECT=multi-agent-turtlesim
```

1. **Run the app as usual** (`pixi run agent`). That's it — LangChain detects these variables automatically and starts sending traces.

To turn tracing off, set `LANGSMITH_TRACING=false` (or remove the lines).

## Reading a trace

Open [smith.langchain.com](https://smith.langchain.com), pick the **multi-agent-turtlesim** project (created automatically on the first run).
You'll see one **trace** per LLM interaction. Click one to open it.

A trace is a tree — outer steps contain inner steps. Click any row to see its full input and output. Things worth looking at:

- **The supervisor's turn** — the system prompt, your message, and which tool it decided to call (e.g. `start_drawing` or `move_forward_tool`).
- **Planner calls** — the exact context injected into the prompt (live turtle pose, parts done, reviewer feedback) and the structured `MovementPlanOutput` the model returned.
- **Reviewer calls** — the trial PNG the vision model actually saw, and its approve/reject verdict. This is the best place to understand *why* a part was rejected.
- **Latency and tokens** — shown per call and summed per trace (A **trace** is one whole mission turn (e.g. “draw a house”)). Useful to see where the time and money go (usually the planner and reviewer).

## Why bother?

- **Debugging** — when a drawing goes wrong, the trace shows whether the blueprint, the plan, or the review made the bad decision (logs can help too)
- **Prompt tweaking** — see exactly what the model receives, change a system prompt in `src/agents/`, and compare traces before/after.
- **Cost awareness** — check how many tokens a mission really uses and which agent is the expensive one.

