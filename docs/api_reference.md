# API Reference

Complete reference for the public functions, classes, and tools exposed by the
Clinical Evidence Intelligence Pipeline.

---

## Pipeline (`src/pipeline.py`)

### `build_pipeline(checkpointer=None)`

Construct and compile the LangGraph `StateGraph`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpointer` | `BaseCheckpointSaver \| None` | `MemorySaver()` | Persistence layer for state checkpointing |

**Returns:** A compiled `CompiledGraph` runnable.

**Side effect:** Saves an auto-generated PNG diagram to `docs/pipeline_graph.png`.

---

### `run_pipeline(request, max_iterations=3, max_retries_per_phase=2, thread_id="default", config=None)`

Execute the full pipeline synchronously and return the final state.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request` | `str` | — | Natural-language content request |
| `max_iterations` | `int` | `3` | Maximum writing-revision cycles |
| `max_retries_per_phase` | `int` | `2` | Times coordinator retries a failing phase |
| `thread_id` | `str` | `"default"` | Checkpointer thread identifier |
| `config` | `dict \| None` | `None` | Extra LangGraph run config |

**Returns:** `dict` — the final `PipelineState`. Key fields to inspect:

| Field | Description |
|-------|-------------|
| `draft_report` | Final markdown report (populated on success) |
| `quality_score` | Composite quality score `[0, 1]` |
| `is_approved` | `True` when quality threshold was met |
| `out_of_scope` | `True` if the request was rejected at the scope gate |
| `scope_rejection_reason` | Explanation when `out_of_scope=True` |
| `clarification_needed` | `True` if the request was too vague |
| `clarification_question` | Follow-up question for the user |
| `surface_error` | `True` on an unrecoverable pipeline error |
| `pipeline_error_message` | Description of the error |
| `errors` | List of non-fatal errors accumulated during the run |

**Example:**
```python
from src.pipeline import run_pipeline

result = run_pipeline(
    request="Create a clinical evidence synthesis for telemedicine in T2D",
    max_iterations=3,
)

if result.get("out_of_scope"):
    print(f"Rejected: {result['scope_rejection_reason']}")
elif result.get("clarification_needed"):
    print(f"Clarify: {result['clarification_question']}")
else:
    print(result["draft_report"])
    print(f"Quality: {result['quality_score']:.2f} | Approved: {result['is_approved']}")
```

---

### `stream_pipeline(request, max_iterations=3, max_retries_per_phase=2, thread_id="default")`

Stream pipeline events node-by-node.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request` | `str` | — | Natural-language content request |
| `max_iterations` | `int` | `3` | Maximum writing-revision cycles |
| `max_retries_per_phase` | `int` | `2` | Per-phase retry cap for coordinator |
| `thread_id` | `str` | `"default"` | Checkpointer thread identifier |

**Yields:** LangGraph event `dict`s (`stream_mode="updates"`). Because all
agents return to coordinator, events alternate between coordinator and agent
nodes, making the orchestration visible in the stream.

**Example:**
```python
from src.pipeline import stream_pipeline

for event in stream_pipeline("Analyse AI regulation policy trends"):
    for node_name, node_state in event.items():
        print(f"[{node_name}] phase={node_state.get('current_phase')}")
```

---

## State (`src/state/pipeline_state.py`)

### `PipelineState` (TypedDict)

Shared state schema flowing through the LangGraph graph.

| Field | Type | Set by | Description |
|-------|------|--------|-------------|
| `request` | `str` | caller | Original user request |
| `content_type` | `str` | coordinator | `clinical_evidence`, `market_report`, `technology_analysis`, `policy_brief`, `general` |
| `pipeline_phase` | `str` | each agent | Current phase, e.g. `"post_research"` |
| `coordinator_next_action` | `str` | coordinator | Routing target for `coordinator_router` |
| `coordinator_instructions` | `dict` | coordinator | Per-agent task enrichment dict |
| `phase_retry_counts` | `dict` | coordinator | Retry counter per phase |
| `out_of_scope` | `bool` | coordinator | `True` if request rejected at scope gate |
| `scope_rejection_reason` | `str` | coordinator | Explanation for scope rejection |
| `clarification_needed` | `bool` | coordinator | `True` if request was ambiguous |
| `clarification_question` | `str` | coordinator | Follow-up question |
| `research_queries` | `list[str]` | research | Search queries executed |
| `raw_sources` | `list[dict]` | research | Source dicts: `{title, url, snippet, source_type}` |
| `research_summary` | `str` | research | Prose synthesis of gathered research |
| `key_findings` | `list[str]` | analysis | Bullet-point findings |
| `statistical_summary` | `dict` | analysis | Key statistics: `{label: value}` |
| `evidence_quality` | `str` | analysis | `strong` / `moderate` / `weak` / etc. |
| `evidence_grade` | `str` | analysis | GRADE level: A / B / C / D |
| `bias_assessment` | `str` | analysis | `low` / `moderate` / `high` / `unclear` |
| `draft_report` | `str` | writing | Full markdown-formatted report |
| `citations` | `list[str]` | writing | Formatted reference list |
| `iteration_count` | `int` | writing | Number of revision cycles completed |
| `quality_score` | `float` | quality | Composite score `[0, 1]` |
| `quality_feedback` | `list[str]` | quality | Actionable improvement feedback |
| `is_approved` | `bool` | quality | `True` when quality threshold is met |
| `max_iterations` | `int` | caller/coordinator | Hard cap on revision cycles |
| `max_retries_per_phase` | `int` | caller/coordinator | Per-phase retry cap |
| `current_phase` | `str` | coordinator | Human-readable phase label |
| `surface_error` | `bool` | coordinator | `True` on unrecoverable error |
| `pipeline_error_message` | `str` | coordinator | Error description |
| `errors` | `list[str]` | all | Accumulated non-fatal errors |
| `messages` | `list[BaseMessage]` | all | Chronological agent message log |

---

## Agent Nodes (`src/agents/`)

All specialised agent nodes share this signature:

```python
def <agent>_node(state: PipelineState, config: RunnableConfig) -> dict:
    ...
```

They accept the full state and a LangGraph `RunnableConfig`, and return a
partial dict of only the fields they modify.

The Coordinator node is the exception — it takes only `state`:

```python
def coordinator_node(state: PipelineState) -> dict:
    ...
```

### `coordinator_node(state)`
Phase-aware hub. Reads `pipeline_phase` to determine which handler to invoke:
`_handle_init`, `_handle_post_research`, `_handle_post_analysis`,
`_handle_post_writing`, or `_handle_post_quality`.

### `coordinator_router(state) → str`
Routing function for the LangGraph conditional edge after coordinator.

| Return value | Next node |
|-------------|-----------|
| `"research"` | `research` |
| `"analysis"` | `analysis` |
| `"writing"` | `writing` |
| `"quality"` | `quality` |
| `"complete"` | `END` |
| `"out_of_scope"` | `END` |
| `"needs_clarification"` | `END` |
| `"surface_error"` | `END` |

### `research_node(state, config)`
Runs the Research Agent using `run_agent_with_forced_tools`. Writes
`research_queries`, `raw_sources`, `research_summary`, `pipeline_phase`.

### `analysis_node(state, config)`
Runs the Analysis Agent using `create_react_agent`. Writes `key_findings`,
`statistical_summary`, `evidence_quality`, `evidence_grade`, `bias_assessment`,
`pipeline_phase`.

### `writing_node(state, config)`
Runs the Writing Agent using `run_agent_with_forced_tools`. Writes
`draft_report`, `citations`, increments `iteration_count`, sets `pipeline_phase`.

### `quality_node(state, config)`
Runs the Quality Agent using `create_react_agent`. Writes `quality_score`,
`quality_feedback`, `is_approved`, `pipeline_phase`.

---

## Research Tools (`src/tools/search_tools.py`)

### `google_search(query, max_results=6)`
Calls the Google Custom Search JSON API.
- **Env vars required:** `GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`
- **Returns:** Formatted numbered list of search results.

### `web_scrape(url, max_chars=3000)`
Extracts main text content from a webpage using requests + BeautifulSoup.
- **Returns:** Clean plain-text content, truncated to `max_chars`.

### `query_medical_database(query, database="pubmed")`
Queries the NCBI PubMed E-utilities API.
- **Returns:** JSON string with article titles, authors, journal, date, URL.

---

## Analysis Tools (`src/tools/analysis_tools.py`)

### `analyze_evidence(research_json)`
Classifies evidence strength and detects themes from a JSON array of sources.
- **Input:** JSON string — list of `{title, full_text/snippet}` dicts.
- **Returns:** JSON with `source_count`, `evidence_strength`, `themes`, `key_points`, `methodology_types`.

### `calculate_statistics(data_json)`
Computes descriptive + inferential statistics.
- **Input:** JSON list of numbers or `{"values": [...]}`.
- **Returns:** JSON with `mean`, `median`, `std_dev`, `min`, `max`, `count`, `confidence_interval_95`.

---

## Writing Tools (`src/tools/writing_tools.py`)

### `generate_report_from_template(context_json)`
Renders a structured report using the built-in Jinja2 template.
- **Input:** JSON with keys: `title`, `executive_summary`, `introduction`, `methodology`, `key_findings`, `evidence_analysis`, `evidence_quality`, `discussion`, `conclusions`, `limitations`, `citations`, `statistical_summary`.
- **Returns:** Rendered markdown string.

### `format_citation(source_json, style="APA")`
Formats a citation in APA, MLA, or Vancouver style.
- **Input:** JSON with `authors`, `year`, `title`, `journal`, `volume`, `issue`, `pages`, `doi`, `url`.
- **Supported styles:** `"APA"`, `"MLA"`, `"Vancouver"`
- **Returns:** Formatted citation string.

### `extract_markdown_section(markdown_text, section_title)`
Extracts a named section from a markdown document by heading (case-insensitive).
- **Returns:** Section content string, or empty string if the section is not found.

### `create_bar_chart(data_json, title, output_path)`
Generates a bar chart and returns it as a base64 data URI.
- **Input:** JSON `{"labels": [...], "values": [...]}`.
- **Returns:** Base64 PNG data URI string.

### `create_forest_plot(studies_json, outcome_label, output_path)`
Generates a forest plot for meta-analysis visualisation.
- **Input:** JSON array of `{study, effect_size, ci_lower, ci_upper, weight}` dicts.
- **Returns:** Base64 PNG data URI string.

### `create_plotly_chart(chart_spec_json)`
Generates a Plotly chart from a flexible specification. Supports multiple chart
types and multi-series data. Always returns Plotly JSON; also returns a base64
PNG if the `kaleido` package is installed.
- **Supported `chart_type` values:** `"bar"`, `"line"`, `"scatter"`, `"pie"`, `"heatmap"`, `"box"`, `"area"`, `"waterfall"`
- **Input:** JSON with `chart_type`, `title`, `x_label`, `y_label`, `data` (list of series dicts).
- **Returns:** JSON string with `plotly_json` (always) and `png_base64` (if kaleido available).

---

## Quality Tools (`src/tools/quality_tools.py`)

### `check_readability(text)`
Computes multiple readability metrics using `textstat` (falls back to a pure
Python implementation if `textstat` is unavailable).
- **Returns:** JSON with `flesch_reading_ease`, `flesch_kincaid_grade`,
  `gunning_fog_index`, `coleman_liau_index`, `smog_index`,
  `automated_readability_index`, `word_count`, `sentence_count`, `interpretation`.

### `check_completeness(report_text, content_type="general")`
Checks for required sections by heading pattern matching.
- **`content_type`:** `"clinical"` uses a stricter required-section list.
- **Returns:** JSON with `present_sections`, `missing_sections`, `completeness_score`, `issues`.

### `check_grammar(text, max_errors=20)`
Checks grammar using LanguageTool (falls back to heuristics if Java / LT unavailable).
- **Returns:** JSON with `error_count`, `quality` (`excellent` / `good` / `fair` / `needs improvement`), `issues`.

### `check_relevancy(report_text, original_request)`
Scores keyword overlap between the report and the original request.
- **Returns:** JSON with `relevancy_score`, `matched_keywords`, `missing_keywords`, `feedback`.

### `check_medical_claims(report_text, sources_json)`
Verifies that specific medical/clinical claims in the report are supported by the
provided sources. Uses keyword matching and sentence-level analysis.
- **Input:** `sources_json` — JSON array of `{title, snippet/full_text}` dicts.
- **Returns:** JSON with `verified_count`, `unverified_count`, `accuracy_score`,
  `unverified_claims`, `feedback`.

---

## Utilities

### `src/utils/agent_runner.py`

#### `run_agent_with_forced_tools(llm, tools, system_prompt, user_message, agent_name, max_iterations=15)`

Custom ReAct loop that forces the LLM to call at least one tool on the first
turn (`tool_choice="any"`). Used by the Research and Writing agents.

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | `BaseChatModel` | The LLM instance to use |
| `tools` | `list` | LangChain `@tool` objects |
| `system_prompt` | `str` | System message content |
| `user_message` | `str` | User message content |
| `agent_name` | `str` | Name used in log output |
| `max_iterations` | `int` | Maximum tool-call iterations (default 15) |

**Returns:** `{"messages": list[BaseMessage]}` — the message list starting from
the `HumanMessage` (system message excluded).

---

### `src/utils/logging_utils.py`

#### `get_logger(name, level="INFO")`
Returns a Rich-enhanced Python logger.

#### `setup_log_file(path: str)`
Opens a plain-text log file. All subsequent `console.print()` output is tee'd
there in real time.

#### `flush_log_to_file()`
Closes the log file (all content has already been streamed).

#### `log_tool_call(agent_name, tool_name, args) → str`
Prints a formatted `→ TOOL` line to the console and returns the args preview.

#### `log_tool_result(agent_name, tool_name, result) → str`
Prints a formatted `← RESULT` line to the console and returns the result preview.

#### `invoke_agent_with_tool_logging(agent, input_messages, config, agent_name) → dict`
Invokes a `create_react_agent` instance and logs every tool call and result.
Used by the Analysis and Quality agents.

#### `PipelineLogger`

High-level observability helper used by all agent nodes.

| Method | Description |
|--------|-------------|
| `phase_start(phase, details="")` | Prints a cyan panel at phase start; records elapsed-time reference |
| `phase_end(phase, summary="")` | Prints a green panel with elapsed time |
| `phase_error(phase, error)` | Prints a red panel for errors |
| `tool_call(tool_name, args)` | Debug-level tool call log |
| `tool_result(tool_name, result_preview)` | Debug-level tool result log |
| `agent_decision(agent, decision, reason="")` | Info-level decision log |
| `coordinator_dispatch(target_agent, instructions)` | Blue panel showing full dispatch instructions |
| `content_preview(label, content, max_chars=600)` | Yellow panel with truncated content preview |
| `quality_result(score, approved, feedback)` | Magenta quality assessment panel |

#### `ToolLoggingCallback`

LangChain `BaseCallbackHandler` that logs tool start / end / error events to
the Rich console. Attached to the Analysis and Quality agents via their
`create_react_agent` config.

---

### `src/utils/token_tracker.py`

#### `TokenTrackingCallback`

LangChain `BaseCallbackHandler` that accumulates token usage per agent and
estimates cost. Used by the FastAPI server (`src/api/routes.py`).

| Method | Description |
|--------|-------------|
| `get_summary() → dict` | Returns per-agent token counts and estimated USD cost |

---

### `src/utils/streaming_callback.py`

#### `ToolStreamingCallback`

LangChain `BaseCallbackHandler` that emits real-time `tool_call` SSE events
as tools start and finish. Used by the FastAPI server for live browser-based
progress updates.

| Constructor | Description |
|-------------|-------------|
| `ToolStreamingCallback(queue, loop)` | `queue` — asyncio queue for SSE events; `loop` — running event loop |
