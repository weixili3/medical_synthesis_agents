# Tests

Unit and integration tests for the Clinical Evidence Intelligence Pipeline.

## Running Tests

From the project root (`research_agents/`):

```bash
# Run all tests
python -m pytest tests/

# Run a specific file
python -m pytest tests/test_tools.py
python -m pytest tests/test_agents.py
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_utils.py

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_agents.py::TestWritingNode -v

# Run a single test
python -m pytest tests/test_agents.py::TestWritingNode::test_successful_writing_updates_state -v
```

> No `GOOGLE_API_KEY` is required — all LLM and API calls are mocked.

---

## Test Files

### `test_tools.py` — Tool Unit Tests

Tests every `@tool`-decorated function in isolation. All external I/O (HTTP, LLM) is mocked.

| Class | Tools Covered |
|---|---|
| `TestGoogleSearch` | `google_search` — credential check, results formatting, empty results |
| `TestWebScrape` | `web_scrape` — content extraction, truncation, timeout handling |
| `TestCalculateStatistics` | `calculate_statistics` — basic stats, dict input, empty/invalid JSON |
| `TestAnalyzeEvidence` | `analyze_evidence` — evidence classification, required schema keys |
| `TestFormatCitation` | `format_citation` — APA, Vancouver, MLA styles, unsupported style error |
| `TestExtractMarkdownSection` | `extract_markdown_section` — section extraction, missing section handling |
| `TestCreatePlotlyChart` | `create_plotly_chart` — bar/line/pie/heatmap/box/area/waterfall types, multi-series, error bars, layout overrides, unsupported type, invalid JSON |
| `TestCheckCompleteness` | `check_completeness` — full/empty/minimal reports, clinical vs general |
| `TestCheckRelevancy` | `check_relevancy` — relevant/irrelevant scoring, required keys |
| `TestCheckReadability` | `check_readability` — metric keys, short-text error, pure-Python fallback |
| `TestCheckGrammar` | `check_grammar` — heuristic fallback, required keys, max_errors cap |
| `TestCheckMedicalClaims` | `check_medical_claims` — claim verification, empty sources, invalid JSON |

---

### `test_agents.py` — Agent Node Unit Tests

Tests each LangGraph agent node function with mocked LLMs and tools.

| Class | Agent Covered |
|---|---|
| `TestCoordinatorNode` | `coordinator_node` — init, post-research, post-analysis, post-quality routing, retry logic |
| `TestCoordinatorRouter` | `coordinator_router` — all routing targets |
| `TestResearchNode` | `research_node` — success, error, coordinator brief injection, JSON fallback |
| `TestAnalysisNode` | `analysis_node` — success, empty input, error handling |
| `TestWritingNode` | `writing_node` — success, error/fallback, JSON fallback, iteration count |
| `TestQualityNode` | `quality_node` — approved, low score, empty draft, JSON parse failure |
| `TestQualityRouter` | `quality_router` — approved, needs research, needs revision |

**Key mocking patterns:**

- Research/Analysis/Quality agents use `create_react_agent` → patch `src.agents.<module>.create_react_agent`
- Writing agent uses `run_agent_with_forced_tools` → patch `src.agents.writing_agent.run_agent_with_forced_tools`
- Quality agent's tool logging uses `invoke_agent_with_tool_logging` → patch `src.agents.quality_agent.invoke_agent_with_tool_logging`

---

### `test_pipeline.py` — Pipeline Integration Tests

Tests the full LangGraph graph with all agents mocked end-to-end.

| Class | What's Tested |
|---|---|
| `TestPipelineBuild` | Graph compiles without error; all 5 nodes present |
| `TestPipelineRunIntegration` | Full research→analysis→writing→quality flow; out-of-scope early exit |
| `TestPipelineStateFlow` | Coordinator output contains required fields for downstream nodes; stale state is cleared on init |

---

### `test_utils.py` — Utility Module Tests

Tests `src/utils/agent_runner.py` and helper functions in `src/utils/logging_utils.py` and `src/tools/quality_tools.py`.

| Class | What's Tested |
|---|---|
| `TestRunAgentWithForcedTools` | Forced first-turn tool choice, tool invocation, unknown tool error, max_iterations cap, multi-tool turns |
| `TestPipelineLoggerMethods` | `coordinator_dispatch`, `content_preview`, `quality_result`, `phase_start/end/error` |
| `TestCountSyllables` | Syllable counting heuristic — single/multi-syllable, silent-e, punctuation stripping |
| `TestPythonReadability` | Pure-Python readability metrics — required keys, value ranges, edge cases |

---

## Coverage Summary

| Module | Tests |
|---|---|
| `src/tools/search_tools.py` | `TestGoogleSearch`, `TestWebScrape` |
| `src/tools/analysis_tools.py` | `TestCalculateStatistics`, `TestAnalyzeEvidence` |
| `src/tools/writing_tools.py` | `TestFormatCitation`, `TestExtractMarkdownSection`, `TestCreatePlotlyChart` |
| `src/tools/quality_tools.py` | `TestCheckCompleteness`, `TestCheckRelevancy`, `TestCheckReadability`, `TestCheckGrammar`, `TestCheckMedicalClaims`, `TestCountSyllables`, `TestPythonReadability` |
| `src/agents/coordinator.py` | `TestCoordinatorNode`, `TestCoordinatorRouter` |
| `src/agents/research_agent.py` | `TestResearchNode` |
| `src/agents/analysis_agent.py` | `TestAnalysisNode` |
| `src/agents/writing_agent.py` | `TestWritingNode` |
| `src/agents/quality_agent.py` | `TestQualityNode`, `TestQualityRouter` |
| `src/utils/agent_runner.py` | `TestRunAgentWithForcedTools` |
| `src/utils/logging_utils.py` | `TestPipelineLoggerMethods` |
| `src/pipeline.py` | `TestPipelineBuild`, `TestPipelineRunIntegration`, `TestPipelineStateFlow` |
