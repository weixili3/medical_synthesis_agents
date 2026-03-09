# Clinical Evidence Intelligence Pipeline

A production-ready multi-agent orchestration system that processes content
intelligence requests end-to-end — from natural-language input to a
fully-reviewed, publication-quality markdown report. Built on **LangGraph**
with **Google Gemini 2.0 Flash** as the language model backbone.

---

## Project Overview

The pipeline accepts a natural-language content request such as:

> *"Create a comprehensive clinical evidence synthesis for the effectiveness of
> telemedicine interventions in managing Type 2 diabetes"*

Five specialised agents collaborate through a shared state graph to deliver:

1. **Scope validation** — rejects off-topic requests before any LLM calls are made
2. **Comprehensive web research** — Google Search, PubMed, and web scraping
3. **Structured data analysis** — evidence classification, statistical summaries
4. **A formatted, cited markdown report** — Jinja2 template, APA / MLA / Vancouver citations
5. **Automated quality review** — readability, completeness, grammar, relevancy, medical claim verification
6. **Iterative revision** — the coordinator re-routes until the quality threshold is met

---

## Architecture

The pipeline uses a **hub-and-spoke topology**: every specialised agent always
returns control to the Coordinator, which then decides the next action based on
the agent's output. This gives the Coordinator full visibility and control at
every step, enabling retries, degraded-path handling, and clean termination.

```
                         START
                           │
                           ▼
              ┌────── coordinator ──────┐
              │       (hub)             │
              │  ┌────────────────────┐ │
              │  │ pipeline_phase?    │ │
              │  │                    │ │
              │  │ init     ──► scope │ │
              │  │ gate/route         │ │
              │  └────────────────────┘ │
              │                         │
   ┌──────────┤ out_of_scope ──► END    │
   │          │ clarification ──► END   │
   │          │ surface_error ──► END   │
   │          │ complete      ──► END   │
   │          └─────────────────────────┘
   │
   │  coordinator dispatches to agents:
   │
   │    ┌──────────────────────────────────────────────────────┐
   │    │  research  ──► Research Agent ──► coordinator        │
   │    │  analysis  ──► Analysis Agent ──► coordinator        │
   │    │  writing   ──► Writing Agent  ──► coordinator        │
   │    │  quality   ──► Quality Agent  ──► coordinator        │
   │    └──────────────────────────────────────────────────────┘
   │
   └─────────────────────────────────────────────────────────► …
```

> See [docs/pipeline_graph.png](docs/pipeline_graph.png) for the compiled
> LangGraph graph diagram, which is auto-generated each time the pipeline is built.

For a detailed walkthrough of the state flow and agent interactions, see
[docs/agent_interactions.md](docs/agent_interactions.md).

For architectural decisions and component design, see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Agent Descriptions

| Agent | Role | Tools |
|-------|------|-------|
| **Coordinator** | Phase-aware orchestrator — validates each agent's output, enriches instructions for the next agent, handles retries, early exits, and routing | — |
| **Research** | Searches Google, PubMed, and web pages; returns structured sources and a prose summary | `google_search`, `web_scrape`, `query_medical_database` |
| **Analysis** | Classifies evidence strength, extracts statistics and themes from research | `analyze_evidence`, `calculate_statistics` |
| **Writing** | Renders a structured markdown report; formats citations; revises drafts based on quality feedback | `generate_report_from_template`, `format_citation`, `extract_markdown_section`, `create_bar_chart`, `create_forest_plot`, `create_plotly_chart` |
| **Quality** | Scores readability, completeness, grammar, relevancy, and medical claim accuracy; approves or rejects the draft | `check_readability`, `check_completeness`, `check_grammar`, `check_relevancy`, `check_medical_claims` |

---

## Framework Justification

**LangGraph** was chosen over alternatives (LangChain AgentExecutor, AutoGen,
CrewAI) for the following reasons:

| Criterion | LangGraph | Reason |
|-----------|-----------|--------|
| **State management** | Explicit `TypedDict` state shared across nodes | Precise control over inter-agent data flow with no hidden coupling |
| **Conditional routing** | Native `add_conditional_edges` | Clean hub-and-spoke orchestration; coordinator decides routing without custom event loops |
| **Streaming** | Built-in `stream_mode="updates"` | Real-time observability with zero extra code |
| **Checkpointing** | `MemorySaver` / persistent backends | Human-in-the-loop and multi-turn resumption out of the box |
| **Debuggability** | LangGraph Studio integration + auto-generated graph diagram | Visual graph inspection and step-through replay |
| **Composability** | Subgraph support | Each agent can be a full inner graph for further specialisation |

**Google Gemini 2.0 Flash** was selected for its strong instruction-following,
reliable JSON output, and low latency — critical for a multi-agent pipeline
where every node makes multiple LLM calls. The model is configurable via
`GEMINI_MODEL` in `.env`.

**`run_agent_with_forced_tools`** (in `src/utils/agent_runner.py`) is used for
the Research and Writing agents to guarantee that the LLM calls at least one
tool on the first turn (`tool_choice="any"`), preventing the agent from
producing a direct response before gathering data.

---

## Frontend

The project includes a React + TypeScript web UI that provides a real-time dashboard for submitting research requests and watching the pipeline run.

**Stack:** React 18, TypeScript, Vite, Tailwind CSS, `react-markdown`

**Key components:**

| Component | Description |
|-----------|-------------|
| `QueryInput` | Text area for submitting a clinical research question |
| `AgentCard` | Live status card per agent (idle → running → complete), with tool call log |
| `MetricsPanel` | Right-panel showing quality score, evidence grade, and token usage |
| `FinalReport` | Rendered markdown report with citation list |

**How it communicates with the backend:**

1. On submit, the UI POSTs to `POST /api/run` → receives a `thread_id`.
2. It immediately opens a Server-Sent Events connection at `GET /api/stream/{thread_id}`.
3. The backend pushes `update`, `tool_call`, `pipeline_complete`, and `pipeline_error` events in real time.
4. The UI state machine (React `useReducer`) transitions through `idle → running → complete / rejected / error`.

In development, Vite proxies all `/api` requests to `http://localhost:8000`, so you only need to run the two servers below.

---

## Setup Instructions

### Prerequisites

- Python 3.11 or later
- Node.js 18+ and npm (for the frontend)
- A Google Cloud project with:
  - **Gemini API** enabled (`GOOGLE_API_KEY`) — free tier available
  - **Custom Search JSON API** enabled (`GOOGLE_SEARCH_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID`)
- *(Optional)* Java 8+ for LanguageTool grammar checking (falls back to heuristics without it)

### 1. Clone the repository

```bash
git clone <repo-url>
cd research_agents
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root by copying the template:

```bash
cp .env.example .env   # if available, otherwise create manually
```

Then edit `.env` with your credentials:

```dotenv
# ── Gemini / LLM ────────────────────────────────────────────────────────────
# Get your API key from https://aistudio.google.com/apikey (free tier available)
GOOGLE_API_KEY=your_gemini_api_key_here

# Gemini model to use — gemini-2.0-flash is the default (fast + cost-efficient)
GEMINI_MODEL=gemini-2.0-flash

# ── Google Custom Search ─────────────────────────────────────────────────────
# 1. In Google Cloud Console, enable "Custom Search JSON API"
# 2. Create an API key under APIs & Services → Credentials
GOOGLE_SEARCH_API_KEY=your_google_cloud_api_key_here

# 1. Go to https://programmablesearchengine.google.com/
# 2. Create a search engine (set "Search the entire web" = ON)
# 3. Copy the CX ID from the control panel
GOOGLE_SEARCH_ENGINE_ID=your_cx_id_here

# ── Pipeline Configuration ───────────────────────────────────────────────────
# Maximum quality-revision cycles before the coordinator forces completion
MAX_ITERATIONS=3

# Python logging level: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO

# Directory where saved reports are written (created automatically)
OUTPUT_DIR=./outputs

# ── FastAPI (only needed if running the REST API server) ─────────────────────
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Where to obtain each credential:**

| Variable | Service | How to get it |
|----------|---------|---------------|
| `GOOGLE_API_KEY` | Google AI Studio | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free tier includes generous quota |
| `GOOGLE_SEARCH_API_KEY` | Google Cloud Console | Enable "Custom Search JSON API" → Create an API key |
| `GOOGLE_SEARCH_ENGINE_ID` | Programmable Search Engine | Create an engine at [programmablesearchengine.google.com](https://programmablesearchengine.google.com/), copy the **CX ID** |

> **Note:** Only `GOOGLE_API_KEY` is strictly required to run the pipeline.
> Without `GOOGLE_SEARCH_API_KEY` / `GOOGLE_SEARCH_ENGINE_ID`, the
> `google_search` tool will return a configuration error (other tools still work).

### 5. Verify installation

```bash
python -c "from src.pipeline import build_pipeline; print('OK')"
```

---

## Running the Application

### Development mode (backend + frontend separately)

**Terminal 1 — FastAPI backend (port 8000):**

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Vite dev server (port 3000):**

```bash
cd src/frontend
npm install        # first time only
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.
The Vite dev server proxies all `/api` requests to the FastAPI backend automatically.

### Production mode (single server)

Build the frontend once, then serve everything from FastAPI:

```bash
cd src/frontend
npm install
npm run build          # outputs to src/frontend/dist/
cd ../..
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

FastAPI automatically detects and serves the compiled static files from `src/frontend/dist/` at `/`, so the entire app is available at [http://localhost:8000](http://localhost:8000).

### API endpoints (standalone / programmatic use)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/run` | Start a pipeline run; returns `{ thread_id }` |
| `GET` | `/api/stream/{thread_id}` | SSE stream of real-time events |
| `GET` | `/api/result/{thread_id}` | Fetch the final result after completion |
| `GET` | `/api/health` | Liveness check |

---

## Usage Examples (CLI)

### Run the primary clinical evidence example

```bash
python main.py "Create a comprehensive clinical evidence synthesis for the effectiveness of telemedicine interventions in managing Type 2 diabetes"
```

### Save the report to a file

```bash
python main.py \
  "Analyse current AI regulation policy trends in the European Union" \
  --output outputs/ai_policy_report.md
```

### Stream node-by-node progress

```bash
python main.py --stream \
  "Market analysis of the global wearable health technology sector 2025"
```

### Print the full pipeline state as JSON

```bash
python main.py --json \
  "Technology analysis of large language model infrastructure providers"
```

### Interactive mode

```bash
python main.py
# > Enter your content request at the prompt
```

### Python API

```python
from src.pipeline import run_pipeline, stream_pipeline

# Synchronous — returns the final state dict
result = run_pipeline(
    request="Policy brief on carbon capture and storage regulation",
    max_iterations=3,
    max_retries_per_phase=2,
)
print(result["draft_report"])
print(f"Quality: {result['quality_score']:.2f} | Approved: {result['is_approved']}")

# Early-exit cases
if result.get("out_of_scope"):
    print(f"Rejected: {result['scope_rejection_reason']}")
elif result.get("clarification_needed"):
    print(f"Please clarify: {result['clarification_question']}")

# Streaming
for event in stream_pipeline("Market report on electric vehicle battery supply chains"):
    for node, state in event.items():
        print(f"[{node}] phase={state.get('current_phase')}")
```

---

## Configuration

### `config/pipeline.yaml`

Top-level pipeline settings: `max_iterations`, `log_level`, `output_dir`,
LLM model selection, routing thresholds, and observability toggles.

### `config/agents.yaml`

Per-agent model, temperature, tool list, and domain-specific settings
(e.g., citation style, minimum key findings, approval thresholds).

Environment variables in `.env` take precedence over YAML for sensitive values.

---

## Running Tests

No API keys are required — all LLM and external API calls are mocked.

```bash
# Run all tests (127 tests)
python -m pytest tests/ -v

# Run only tool tests
python -m pytest tests/test_tools.py -v

# Run only agent tests
python -m pytest tests/test_agents.py -v

# Run pipeline integration tests
python -m pytest tests/test_pipeline.py -v

# Run utility tests
python -m pytest tests/test_utils.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

See [tests/README.md](tests/README.md) for the full test coverage breakdown and
mocking patterns used.

---

## Logs

The pipeline writes a structured log file to:

```
examples/logs/YYYY-MM-DD/HH-MM-SS.log
```

The log file captures all Rich console output (plain text, no ANSI colours),
including phase banners, tool calls, coordinator decisions, and quality scores.
This is streamed in real time — there is no buffering.

---

## Performance Considerations

| Concern | Approach |
|---------|----------|
| **LLM latency** | Gemini 2.0 Flash chosen for low per-token latency; configurable via `GEMINI_MODEL` |
| **Forced tool use** | `run_agent_with_forced_tools` uses `tool_choice="any"` on the first turn to prevent the LLM from skipping tools |
| **Tool timeouts** | All HTTP calls have explicit `timeout=15s`; failures return descriptive strings so the LLM can handle them gracefully |
| **Context length** | Research summaries and reports are truncated before inclusion in prompts |
| **Revision loops** | Hard cap via `max_iterations` prevents runaway loops; coordinator auto-completes when the limit is reached |
| **Phase retries** | `max_retries_per_phase` (default 2) allows the coordinator to retry a failing agent before degrading gracefully |
| **Parallelism** | LangGraph supports parallel node execution; future work can fan out research queries concurrently |
| **Checkpointing** | `MemorySaver` is in-process; swap for `SqliteSaver` or `PostgresSaver` for production persistence |

---

## Repository Structure

```
research_agents/
├── README.md                    # This file
├── ARCHITECTURE.md              # System design and architectural decisions
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (never commit this)
├── main.py                      # CLI entry point
├── src/
│   ├── pipeline.py              # LangGraph graph assembly + run/stream helpers
│   ├── state/
│   │   └── pipeline_state.py    # PipelineState TypedDict
│   ├── agents/
│   │   ├── coordinator.py       # Phase-aware hub coordinator node + router
│   │   ├── scope_gate.py        # Keyword-based out-of-scope detection
│   │   ├── research_agent.py    # Research node (run_agent_with_forced_tools)
│   │   ├── analysis_agent.py    # Analysis node (create_react_agent)
│   │   ├── writing_agent.py     # Writing node (run_agent_with_forced_tools)
│   │   └── quality_agent.py     # Quality node + router (create_react_agent)
│   ├── tools/
│   │   ├── search_tools.py      # google_search, web_scrape, query_medical_database
│   │   ├── analysis_tools.py    # analyze_evidence, calculate_statistics
│   │   ├── writing_tools.py     # generate_report_from_template, format_citation,
│   │   │                        # extract_markdown_section, create_bar_chart, create_forest_plot
│   │   └── quality_tools.py     # check_readability, check_completeness, check_grammar,
│   │                            # check_relevancy, check_medical_claims
│   ├── utils/
│   │   ├── logging_utils.py     # PipelineLogger, ToolLoggingCallback, Rich console
│   │   ├── agent_runner.py      # run_agent_with_forced_tools (forced first-turn tool call)
│   │   ├── streaming_callback.py# ToolStreamingCallback for SSE streaming (API server)
│   │   └── token_tracker.py     # TokenTrackingCallback for usage/cost tracking (API server)
│   ├── api/
│   │   ├── main.py              # FastAPI app — CORS, static file serving
│   │   ├── routes.py            # REST + SSE streaming endpoints
│   │   └── models.py            # Pydantic request/response models
│   └── frontend/                # React + TypeScript web UI
│       ├── src/
│       │   ├── App.tsx          # Root component, SSE subscription, reducer
│       │   ├── types.ts         # Shared TypeScript types
│       │   ├── api/client.ts    # startRun() and openStream() helpers
│       │   └── components/
│       │       ├── QueryInput.tsx    # Research question input form
│       │       ├── AgentCard.tsx     # Per-agent status + tool call log
│       │       ├── MetricsPanel.tsx  # Quality score, evidence grade, tokens
│       │       └── FinalReport.tsx   # Rendered markdown report + citations
│       ├── package.json         # npm dependencies (React, Vite, Tailwind)
│       ├── vite.config.ts       # Dev server on :3000, proxies /api → :8000
│       └── dist/                # Built output (created by npm run build)
├── tests/
│   ├── README.md                # Test coverage breakdown and run instructions
│   ├── test_tools.py            # Unit tests for all tool functions (mocked I/O)
│   ├── test_agents.py           # Unit tests for all agent nodes (mocked LLM)
│   ├── test_pipeline.py         # Integration tests for the full graph
│   └── test_utils.py            # Unit tests for agent_runner and logging_utils
├── config/
│   ├── agents.yaml              # Per-agent configuration
│   └── pipeline.yaml            # Pipeline-level configuration
├── docs/
│   ├── pipeline_graph.png       # Auto-generated LangGraph diagram
│   ├── agent_interactions.md    # Inter-agent communication patterns
│   └── api_reference.md         # Full public API reference
└── examples/
    ├── clinical_evidence_request.json  # Sample request with expected metadata
    ├── clinical-evidence-report-example.md
    └── logs/                    # Pipeline run logs (YYYY-MM-DD/HH-MM-SS.log)
```

---

## Future Enhancements

- **Parallel research fan-out** — run multiple search queries concurrently using
  LangGraph's parallel node execution to cut research latency significantly.
- **Persistent checkpointing** — replace `MemorySaver` with `PostgresSaver` for
  durable state storage and cross-session resumption (partial output recovery).
- **Human-in-the-loop** — add a coordinator interrupt after the Writing Agent
  to allow domain experts to annotate the draft before quality scoring.
- **RAG with vector store** — integrate a Chroma or Pinecone vector store to
  index scraped documents and enable semantic retrieval within the pipeline.
- **Citation verification** — add a DOI-resolution tool to validate that cited
  sources are real, accessible, and correctly attributed.
- **Multi-format output** — extend the Writing Agent to export PDF (via
  `weasyprint`) and DOCX (via `python-docx`) in addition to markdown.
- **Evaluation harness** — build a benchmarking suite using labelled requests
  and expected outputs to track quality regression across model versions.
- **Async pipeline** — convert node functions to `async` for non-blocking I/O,
  enabling higher throughput when running multiple pipelines concurrently.
