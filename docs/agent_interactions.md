# Agent Interaction Patterns

This document describes how the five agents in the Clinical Evidence Intelligence
Pipeline communicate, share state, and handle control flow.

---

## 1. Communication Model

Agents do **not** communicate directly with one another. All inter-agent
communication happens through the **shared `PipelineState`** object managed by
LangGraph. Each node reads the fields it needs, performs its work, and writes
only its own output fields back.

```
User Request
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│                     PipelineState (shared)                   │
│  request · content_type · research_summary · key_findings    │
│  draft_report · quality_score · is_approved · messages · … │
└──────────┬───────────────────────────────────────────────────┘
           │  read / write (per node)
     ┌─────┴──────┐
     │            │
  Agent A      Agent B      (never talk directly)
```

Each agent signals completion by setting `pipeline_phase` to a value like
`"post_research"` or `"post_writing"`. The Coordinator reads this on the next
turn to know which handler to invoke.

---

## 2. Hub-and-Spoke Workflow

All agents always return to the Coordinator after completing their work. The
Coordinator is the sole routing authority.

```
START
  │
  ▼
[Coordinator — init]
  reads : request
  does  : scope gate → content-type detection → state initialisation
  writes: content_type, max_iterations, coordinator_instructions, all defaults
  routes: out_of_scope → END
          needs_clarification → END
          valid → research
  │
  ▼
[Research Agent]
  reads : request, coordinator_instructions["research"]
  writes: research_queries, raw_sources, research_summary
          pipeline_phase = "post_research"
  │
  ▼
[Coordinator — post_research]
  validates: source count, summary length
  writes   : coordinator_instructions["analysis"]
  routes   : retry → research (up to max_retries_per_phase)
             proceed → analysis
  │
  ▼
[Analysis Agent]
  reads : research_summary, raw_sources, coordinator_instructions["analysis"]
  writes: key_findings, statistical_summary, evidence_quality, evidence_grade
          pipeline_phase = "post_analysis"
  │
  ▼
[Coordinator — post_analysis]
  validates: key findings count
  writes   : coordinator_instructions["writing"]
  routes   : retry → analysis
             proceed → writing
  │
  ▼
[Writing Agent]
  reads : request, key_findings, research_summary, evidence_quality,
          statistical_summary, raw_sources, coordinator_instructions["writing"]
  writes: draft_report, citations, iteration_count (incremented)
          pipeline_phase = "post_writing"
  │
  ▼
[Coordinator — post_writing]
  validates: draft length
  writes   : coordinator_instructions["quality"]
  routes   : retry → writing
             proceed → quality
  │
  ▼
[Quality Agent]
  reads : draft_report, request, content_type, iteration_count,
          coordinator_instructions["quality"]
  writes: quality_score, quality_feedback, is_approved
          pipeline_phase = "post_quality"
  │
  ▼
[Coordinator — post_quality]
  reads : is_approved, quality_feedback, iteration_count
  routes: is_approved = True ─────────────────────────► complete → END
          prose/structure issue ──────────────────────► writing
          missing evidence / insufficient data ───────► research
          iteration_count >= max_iterations ──────────► complete → END
```

---

## 3. Coordinator Instructions

The Coordinator enriches each agent's task through `coordinator_instructions`,
a dict keyed by agent name:

```python
state["coordinator_instructions"] = {
    "research": "Focus on RCTs published after 2020. The quality review noted...",
    "analysis": "Prioritise HbA1c outcomes and patient adherence metrics.",
    "writing":  "Expand the Discussion section. Address the limitation on...",
    "quality":  "Review 9638-char draft. Check readability, completeness...",
}
```

Each agent appends these instructions to its user message prompt so the LLM
has targeted guidance beyond the generic system prompt.

---

## 4. Revision Loop

When the Quality Agent rejects a draft, the Coordinator reads `quality_feedback`
to decide where to route:

| Condition | Destination | Rationale |
|-----------|-------------|-----------|
| Feedback mentions "missing", "insufficient data", "lacks sources" | Research Agent | Fetch additional evidence |
| Any other quality issue (readability, structure, grammar) | Writing Agent | Revise the existing draft |
| `iteration_count >= max_iterations` | `complete` → END | Prevent infinite loops |

The `iteration_count` is incremented by the Writing Agent on each run.

---

## 5. State Fields by Agent

### Coordinator
| Field | Action |
|-------|--------|
| `request` | Reads |
| `pipeline_phase` | Reads (determines which handler to invoke) |
| `content_type` | Writes (detected from keywords + LLM) |
| `max_iterations`, `max_retries_per_phase` | Reads from state or sets defaults |
| `coordinator_instructions` | Writes (per-agent task enrichment) |
| `coordinator_next_action` | Writes (drives LangGraph routing) |
| `out_of_scope`, `clarification_needed` | Writes (early-exit flags) |
| All other fields | Writes safe defaults on `init` |

### Research Agent
| Field | Action |
|-------|--------|
| `request` | Reads |
| `coordinator_instructions["research"]` | Reads (targeted brief) |
| `research_queries` | Writes |
| `raw_sources` | Writes |
| `research_summary` | Writes |
| `pipeline_phase` | Writes → `"post_research"` |

### Analysis Agent
| Field | Action |
|-------|--------|
| `research_summary` | Reads |
| `raw_sources` | Reads |
| `request`, `coordinator_instructions["analysis"]` | Reads |
| `key_findings` | Writes |
| `statistical_summary` | Writes |
| `evidence_quality`, `evidence_grade`, `bias_assessment` | Writes |
| `pipeline_phase` | Writes → `"post_analysis"` |

### Writing Agent
| Field | Action |
|-------|--------|
| `request`, `key_findings`, `research_summary` | Reads |
| `evidence_quality`, `statistical_summary`, `raw_sources` | Reads |
| `coordinator_instructions["writing"]`, `content_type` | Reads |
| `draft_report` | Writes |
| `citations` | Writes |
| `iteration_count` | Increments |
| `pipeline_phase` | Writes → `"post_writing"` |

### Quality Agent
| Field | Action |
|-------|--------|
| `draft_report`, `request`, `content_type` | Reads |
| `raw_sources` (for claim verification) | Reads |
| `iteration_count`, `max_iterations` | Reads |
| `coordinator_instructions["quality"]` | Reads |
| `quality_score` | Writes |
| `quality_feedback` | Writes |
| `is_approved` | Writes |
| `pipeline_phase` | Writes → `"post_quality"` |

---

## 6. Tool Usage per Agent

```
Research Agent   (run_agent_with_forced_tools — tool call forced on turn 1)
  ├── google_search          → Google Custom Search JSON API
  ├── web_scrape             → requests + BeautifulSoup
  └── query_medical_database → NCBI PubMed E-utilities

Analysis Agent   (create_react_agent)
  ├── analyze_evidence       → keyword-based evidence classifier
  └── calculate_statistics   → numpy / scipy descriptive + inferential stats

Writing Agent    (run_agent_with_forced_tools — tool call forced on turn 1)
  ├── generate_report_from_template → Jinja2 markdown renderer
  ├── format_citation               → APA / MLA / Vancouver formatter
  ├── extract_markdown_section      → regex section extractor
  ├── create_bar_chart              → matplotlib bar chart → base64 PNG
  ├── create_forest_plot            → matplotlib forest plot → base64 PNG
  └── create_plotly_chart           → Plotly interactive chart → JSON + optional PNG

Quality Agent    (create_react_agent)
  ├── check_readability    → textstat (Flesch, FK Grade, Gunning Fog, etc.)
  ├── check_completeness   → required-section presence checker
  ├── check_grammar        → LanguageTool / heuristic fallback
  ├── check_relevancy      → keyword-overlap scorer
  └── check_medical_claims → claim-to-source verification
```

---

## 7. Error Handling

| Scenario | Handling |
|----------|----------|
| Tool raises an exception | Tool catches it; returns a descriptive error string; LLM decides how to proceed |
| Agent output is not valid JSON | Parser falls back to raw text or safe defaults; warning logged |
| Agent node raises an exception | Error appended to `state["errors"]`; safe defaults written; `pipeline_phase` still set so Coordinator can handle it |
| Phase fails repeatedly | Coordinator retries up to `max_retries_per_phase` then routes to next phase with degraded data |
| Max iterations reached | Coordinator auto-completes; draft is returned regardless of quality score |

All errors accumulate in `state["errors"]` and are surfaced in the final output
without aborting the pipeline.

---

## 8. Observability

Every node uses `PipelineLogger` to emit structured output:

- **Phase start / end** banners with elapsed time (Rich panels, cyan/green)
- **Coordinator dispatch** panels (blue) showing the full instruction sent to each agent
- **Tool call / result** lines (cyan/green) for every tool invocation
- **Agent decisions** info logs (routing choices with rationale)
- **Quality result** summary panel (score, approval status, per-dimension feedback)

All agent messages are appended to `state["messages"]` for a full chronological
transcript across the pipeline run.
