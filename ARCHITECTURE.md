# Architecture

This document describes the system design, key decisions, and component
interactions of the Clinical Evidence Intelligence Pipeline.

---

## 1. High-Level Design

The pipeline follows a **hub-and-spoke pattern** implemented as a LangGraph
`StateGraph`. The Coordinator is the sole routing authority — every specialised
agent always returns control to it, and the Coordinator decides the next step
based on the agent's output and the current `pipeline_phase`.

```
                              START
                                │
                                ▼
                         ┌────────────┐
                    ┌───►│ Coordinator │◄──────────────────────────┐
                    │    └─────┬──────┘                            │
                    │          │  coordinator_router()              │
                    │    ┌─────▼──────────────────────────────┐    │
                    │    │ out_of_scope / clarification /      │    │
                    │    │ surface_error / complete  ──► END   │    │
                    │    └─────┬──────────────────────────────-┘    │
                    │          │                                     │
                    │   ┌──────┴──────────────────────────┐         │
                    │   │  dispatch to agent               │         │
                    │   │                                  │         │
                    │   │  research ──► Research Agent ───►│─────────┤
                    │   │  analysis ──► Analysis Agent ───►│─────────┤
                    │   │  writing  ──► Writing Agent  ───►│─────────┤
                    │   │  quality  ──► Quality Agent  ───►│─────────┘
                    │   └──────────────────────────────────┘
                    │
                    └── (all agents always return to coordinator)
```

All agents share a single `PipelineState` TypedDict. Each node reads the fields
it needs, does its work, and returns a **partial dict** of only the fields it
modifies — LangGraph merges these updates into the shared state automatically.

---

## 2. Component Breakdown

### 2.1 PipelineState

`src/state/pipeline_state.py`

A `TypedDict` that acts as the single source of truth for the entire pipeline
run. The `messages` field uses LangGraph's `Annotated[list, add_messages]`
reducer, which appends new messages rather than replacing the list, enabling a
full chronological trace of all agent actions.

Key field groups:

| Group | Fields |
|-------|--------|
| Input | `request`, `max_iterations`, `max_retries_per_phase` |
| Orchestration | `pipeline_phase`, `coordinator_next_action`, `phase_retry_counts`, `coordinator_instructions` |
| Scope gate | `out_of_scope`, `scope_rejection_reason`, `clarification_needed`, `clarification_question` |
| Research | `research_queries`, `raw_sources`, `research_summary` |
| Analysis | `key_findings`, `statistical_summary`, `evidence_quality`, `evidence_grade`, `bias_assessment` |
| Writing | `draft_report`, `citations`, `iteration_count` |
| Quality | `quality_score`, `quality_feedback`, `is_approved` |
| Control | `current_phase`, `errors`, `messages` |

### 2.2 Coordinator Agent

`src/agents/coordinator.py`

The Coordinator is a **pure Python function** — it does not call an LLM on
every turn. It is **phase-aware**: it reads `pipeline_phase` (set by each
specialist agent before returning) to know which agent just reported, then
dispatches to the appropriate handler:

| `pipeline_phase` | Handler |
|-----------------|---------|
| `"init"` | `_handle_init` — scope gate, content-type detection, state initialisation |
| `"post_research"` | `_handle_post_research` — validates sources; enriches analysis instructions |
| `"post_analysis"` | `_handle_post_analysis` — validates findings; enriches writing instructions |
| `"post_writing"` | `_handle_post_writing` — validates draft; enriches quality instructions |
| `"post_quality"` | `_handle_post_quality` — approves, routes to revision, or forces completion |

The `init` phase uses a single LLM call (`_llm_init_check`) to perform scope
validation, clarity checking, and research brief generation. Every subsequent
phase is LLM-free, keeping orchestration latency minimal.

**Scope gate** (`src/agents/scope_gate.py`): keyword-based pre-filter that
rejects clearly off-topic requests before any LLM call is made.

### 2.3 Research Agent

`src/agents/research_agent.py`

- Uses **`run_agent_with_forced_tools`** (not `create_react_agent`) to guarantee
  the LLM calls at least one search tool on the first turn before producing output.
  The first call uses `tool_choice="any"`; subsequent calls in the same turn are
  unconstrained.
- Generates 3–5 targeted search queries, calls Google Search, scrapes key URLs,
  and queries PubMed for clinical topics.
- Expects JSON output (`research_queries`, `raw_sources`, `research_summary`);
  falls back to raw-text summary if JSON parsing fails.
- In revision loops, `coordinator_instructions["research"]` contains a targeted
  brief from the Coordinator based on quality feedback.

### 2.4 Analysis Agent

`src/agents/analysis_agent.py`

- Uses **`create_react_agent`** — analysis is data-driven and does not require
  forced tool use.
- Calls `analyze_evidence` to classify evidence strength and detect themes.
- Calls `calculate_statistics` if the research mentions numerical data.
- Expects JSON output (`key_findings`, `statistical_summary`, `evidence_quality`,
  `evidence_grade`, `bias_assessment`, `methodology_types`, `themes`).

### 2.5 Writing Agent

`src/agents/writing_agent.py`

- Uses **`run_agent_with_forced_tools`** to guarantee the agent calls
  `generate_report_from_template` rather than producing a direct text response.
- Builds a context dict from all upstream state fields.
- Calls `format_citation` (APA by default) for each source in `raw_sources`.
- In revision loops, `coordinator_instructions["writing"]` contains the specific
  improvements requested by the quality review.
- `_fallback_report` generates a minimal structured report without LLM
  involvement if the agent raises an exception.

### 2.6 Quality Agent

`src/agents/quality_agent.py`

- Uses **`create_react_agent`** wrapped in `invoke_agent_with_tool_logging` for
  tool-call observability.
- Computes five sub-scores:

  | Sub-score | Weight | Tool |
  |-----------|--------|------|
  | Completeness | 35% | `check_completeness` |
  | Relevancy | 35% | `check_relevancy` |
  | Readability | 15% | `check_readability` |
  | Grammar | 15% | `check_grammar` |
  | Medical claims | — | `check_medical_claims` (informational) |

- Sets `is_approved = True` when composite score ≥ 0.70 **and** completeness ≥ 0.75.
- If `iteration_count >= max_iterations`, auto-approves to prevent infinite loops.
- Sets `pipeline_phase = "post_quality"` so the Coordinator knows which handler to call.

---

## 3. LangGraph Graph Topology

```python
workflow = StateGraph(PipelineState)

# Nodes
workflow.add_node("coordinator", coordinator_node)
workflow.add_node("research",    research_node)
workflow.add_node("analysis",    analysis_node)
workflow.add_node("writing",     writing_node)
workflow.add_node("quality",     quality_node)

# Entry point
workflow.add_edge(START, "coordinator")

# Coordinator is the sole decision-maker
workflow.add_conditional_edges(
    "coordinator",
    coordinator_router,
    {
        "research":            "research",
        "analysis":            "analysis",
        "writing":             "writing",
        "quality":             "quality",
        "complete":            END,
        "out_of_scope":        END,
        "needs_clarification": END,
        "surface_error":       END,
    },
)

# All agents always report back to coordinator (hub-and-spoke)
workflow.add_edge("research",  "coordinator")
workflow.add_edge("analysis",  "coordinator")
workflow.add_edge("writing",   "coordinator")
workflow.add_edge("quality",   "coordinator")
```

---

## 4. ReAct Agent Patterns

Two agent execution patterns are used, chosen per agent's requirements:

### `create_react_agent` (Analysis, Quality)

Standard LangGraph ReAct loop: the LLM may call tools or produce a final answer
freely on any turn. Suitable when the agent can reason about whether tools are
needed.

```
User message
     │
     ▼
  ┌──────┐
  │  LLM │◄─── System prompt
  └──┬───┘
     │
  Tool calls?
  ├─ Yes → invoke tools → append results → loop back to LLM
  └─ No  → final answer
```

### `run_agent_with_forced_tools` (Research, Writing)

Custom ReAct loop in `src/utils/agent_runner.py`. On the first turn only,
`tool_choice="any"` forces the LLM to call a tool before producing output.
Subsequent turns are unconstrained. This prevents the agents from generating
a report or summary without first gathering data.

```
User message
     │
     ▼
  ┌──────┐
  │  LLM │◄─── System prompt + tool_choice="any" (first turn only)
  └──┬───┘
     │
  Must call tool (first turn)
  ├─ Tool calls → invoke → append → loop (auto mode)
  └─ No tools on subsequent turns → final answer
```

---

## 5. Tool Architecture

Tools are standard LangChain `@tool`-decorated functions grouped by agent:

```
src/tools/
├── search_tools.py   → google_search, web_scrape, query_medical_database
│                        (HTTP via requests + BeautifulSoup + NCBI API)
│
├── analysis_tools.py → analyze_evidence, calculate_statistics
│                        (pure Python + numpy/scipy)
│
├── writing_tools.py  → generate_report_from_template, format_citation,
│                        extract_markdown_section, create_bar_chart,
│                        create_forest_plot, create_plotly_chart
│                        (Jinja2 + matplotlib + Plotly + regex)
│
└── quality_tools.py  → check_readability, check_completeness, check_grammar,
                         check_relevancy, check_medical_claims
                         (textstat + LanguageTool/heuristics + regex)
```

Each tool module exposes a `get_<domain>_tools()` factory that returns the list
of tool objects bound to the corresponding agent.

**Design contract:** Tools **never raise unhandled exceptions** — all failure
paths return descriptive error strings so the LLM can decide how to proceed.

---

## 6. State Flow Diagram

```
                  ┌──────────────────────────────────────┐
                  │            PipelineState              │
                  ├──────────────────────────────────────┤
Coordinator init: │ content_type, max_iterations,        │
                  │ coordinator_instructions, all defaults│
                  ├──────────────────────────────────────┤
Research writes:  │ research_queries, raw_sources,        │
                  │ research_summary, pipeline_phase      │
                  ├──────────────────────────────────────┤
Analysis writes:  │ key_findings, statistical_summary,    │
                  │ evidence_quality, evidence_grade,     │
                  │ bias_assessment, pipeline_phase       │
                  ├──────────────────────────────────────┤
Writing writes:   │ draft_report, citations,              │
                  │ iteration_count (incremented),        │
                  │ pipeline_phase                        │
                  ├──────────────────────────────────────┤
Quality writes:   │ quality_score, quality_feedback,      │
                  │ is_approved, pipeline_phase           │
                  └──────────────────────────────────────┘
```

---

## 7. Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| **Tool** | `try/except` wraps all I/O; returns descriptive string on failure |
| **JSON parsing** | Fallback to raw text or safe defaults; logs a warning |
| **Agent node** | `try/except` around full agent invocation; writes fallback state; appends to `errors` |
| **Phase retry** | Coordinator retries a failing phase up to `max_retries_per_phase` times before degrading |
| **Revision loop** | `max_iterations` hard cap; coordinator forces completion when reached |
| **Missing state fields** | Coordinator initialises all fields on `init`; agents use `.get(field, default)` |
| **Out-of-scope** | Keyword gate in `init` phase exits immediately, no downstream agents called |

---

## 8. Observability

### Rich Console + Log File

`PipelineLogger` (`src/utils/logging_utils.py`) emits structured output at three levels:

- **Rich console panels** — coloured phase start/end banners with elapsed time
- **INFO logs** — phase transitions, coordinator decisions, dispatch instructions, quality scores
- **DEBUG logs** — individual tool calls and result previews

Each pipeline run writes a plain-text log to `examples/logs/YYYY-MM-DD/HH-MM-SS.log`,
streamed in real time (no buffering). This is set up in `main.py` via
`setup_log_file()` from `logging_utils`.

### Token Tracking + SSE Streaming (API Server)

Two additional callback handlers are used by the FastAPI server (`src/api/routes.py`):

- **`TokenTrackingCallback`** (`src/utils/token_tracker.py`) — accumulates input/output
  token counts per agent and estimates cost using Gemini 2.0 Flash pricing.
- **`ToolStreamingCallback`** (`src/utils/streaming_callback.py`) — emits
  real-time `tool_call` SSE events as tools start and finish, enabling
  live progress in browser-based UIs.

### Message Transcript

All agent messages are accumulated in `state["messages"]` via the `add_messages`
reducer, providing a full chronological transcript of every LLM interaction for
post-run auditing.

---

## 9. Configuration Layers

Settings are resolved in this priority order (highest first):

```
1. Environment variables  (.env / shell export)
2. config/pipeline.yaml   (pipeline-level defaults)
3. config/agents.yaml     (per-agent defaults)
4. Hard-coded defaults    (in source code)
```

This allows the same codebase to run in development (with `.env`), CI (with
environment variables injected by the runner), and production (with YAML +
secrets manager).

---

## 10. Frontend Architecture

### 10.1 Overview

The frontend is a single-page application (SPA) built with React 18, TypeScript,
Vite, and Tailwind CSS. It lives in `src/frontend/` and communicates with the
FastAPI backend exclusively through two API calls: a REST POST to start a run
and a Server-Sent Events (SSE) stream to receive real-time updates.

```
Browser
  │
  ├── POST /api/run  ────────────────────────────────► FastAPI
  │        { question }                                 returns { thread_id }
  │
  └── GET  /api/stream/{thread_id}  ─── SSE ──────────► FastAPI
           ← update events
           ← tool_call events
           ← pipeline_complete
           ← pipeline_error
```

In development, Vite's dev server (port 3000) proxies `/api/*` to the FastAPI
backend at port 8000, so the frontend never needs to know the backend URL. In
production, FastAPI serves the compiled `dist/` folder directly (see
`src/api/main.py`), making the entire app a single-origin deployment.

---

### 10.2 State Machine (`App.tsx`)

All application state is managed by a single `useReducer` in `App.tsx`. The
reducer is the only place state transitions happen — components are pure
functions of state.

**State shape (`AppState`):**

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"idle" \| "running" \| "complete" \| "rejected" \| "error"` | Current pipeline lifecycle phase |
| `threadId` | `string \| null` | Active run identifier |
| `agents` | `Record<AgentName, AgentState>` | Per-agent status + tool call log |
| `finalReport` | `string` | Markdown report text |
| `qualityScore` | `number` | 0–1 composite quality score |
| `isApproved` | `boolean` | Whether quality gate passed |
| `tokenSummary` | `TokenSummary \| null` | Live token usage per agent |
| `rejectionMessage` | `string \| null` | Out-of-scope or clarification message |
| `pipelineErrors` | `string[]` | Non-fatal warnings accumulated during run |

**Transitions:**

```
RESET ──────────────────────────────────────────► idle
START (POST /api/run returns thread_id) ────────► running
UPDATE (SSE: coordinator_decision / agent_complete) stays running, mutates agents[]
TOOL_CALL (SSE: tool start/end) ────────────────► appends to agent.toolCalls[]
COMPLETE (SSE: pipeline_complete) ──────────────► complete | rejected | error
ERROR (SSE: pipeline_error / network drop) ─────► error
```

Each agent entry has its own sub-status (`"idle" | "running" | "complete"`).
The coordinator dispatches the next agent by updating `coordinator_next_action`,
which the `UPDATE` reducer uses to set that agent's status to `"running"`.

---

### 10.3 SSE Subscription (`App.tsx` — `useEffect`)

The `useEffect` hook fires whenever `threadId` changes (i.e. once per new run).
It opens a native `EventSource` connection and registers named event listeners:

| SSE event name | Action dispatched | Effect |
|----------------|-------------------|--------|
| `update` | `UPDATE` | Updates agent statuses and token summary |
| `tool_call` | `TOOL_CALL` | Appends or updates a tool call entry in the agent card |
| `pipeline_complete` | `COMPLETE` | Populates final report, metrics, closes stream |
| `pipeline_error` | `ERROR` | Sets error message, closes stream |
| `stream_end` | — | Closes `EventSource` cleanly |

The cleanup function returned by `useEffect` calls `es.close()`, ensuring the
SSE connection is torn down if the component unmounts or the user starts a new
run.

---

### 10.4 Component Breakdown

```
App.tsx  (state, SSE subscription, layout)
│
├── QueryInput.tsx
│     Controlled textarea + submit button. Disabled while status === "running".
│     Calls props.onSubmit(question) → App calls POST /api/run.
│
├── AgentCard.tsx  (rendered once per agent via AGENT_NAMES map)
│     Shows agent name, status badge (idle / running / complete), and a
│     collapsible tool call log. Each tool call entry shows the tool name,
│     input arguments, and output (once available from the "end" phase event).
│
├── MetricsPanel.tsx
│     Right-column panel. Displays quality score (0–100%), evidence grade,
│     approval status, and a per-agent token usage breakdown from tokenSummary.
│     All values update live as SSE events arrive.
│
└── FinalReport.tsx
      Renders draft_report as GitHub-Flavoured Markdown via react-markdown +
      remark-gfm. Displays the citations list below the report body.
      Only mounted when state.finalReport is non-empty.
```

---

### 10.5 API Client (`src/api/client.ts`)

Two thin functions wrap all network calls:

```typescript
// Start a pipeline run; returns the thread_id
startRun({ question, max_iterations?, max_retries_per_phase? }): Promise<string>

// Open an SSE stream for a running pipeline
openStream(threadId: string): EventSource
```

The `BASE = "/api"` constant means all requests are relative to the current
origin — no hardcoded host — which works identically in dev (proxied by Vite)
and production (served by FastAPI).

---

### 10.6 Build and Serving

**Development:**
```
Browser :3000  ──► Vite dev server (HMR)
                       └── /api/* proxy ──► FastAPI :8000
```

**Production:**
```
Browser  ──► FastAPI :8000
               ├── /api/*  ──► APIRouter (routes.py)
               └── /*      ──► StaticFiles(src/frontend/dist/)
```

`src/api/main.py` mounts the `dist/` directory only if it exists, so the
backend starts cleanly without a frontend build (CLI / test environments).

---

## 11. Scalability Considerations

| Dimension | Current | Recommended for Production |
|-----------|---------|---------------------------|
| Checkpointer | `MemorySaver` (in-process) | `PostgresSaver` or `RedisSaver` |
| Concurrency | Single-threaded pipeline runs | Async LangGraph with `asyncio` |
| Research parallelism | Sequential search queries | Parallel fan-out sub-graph |
| LLM provider | Single Gemini model | Model fallback chain (Gemini → OpenAI) |
| Deployment | Python script / FastAPI dev server | FastAPI + Celery workers + Docker |
| Observability | Rich console + Python logging | LangSmith / OpenTelemetry + Grafana |
| Cost tracking | `TokenTrackingCallback` (per-request) | Aggregated metrics store |
