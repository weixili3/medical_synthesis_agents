import { useEffect, useReducer } from "react";
import { Activity, AlertTriangle, MessageSquareWarning } from "lucide-react";
import QueryInput from "./components/QueryInput";
import AgentCard from "./components/AgentCard";
import MetricsPanel from "./components/MetricsPanel";
import FinalReport from "./components/FinalReport";
import { startRun, openStream } from "./api/client";
import type {
  AppState,
  AppAction,
  AgentName,
  UpdateEvent,
  CompletionData,
  TokenSummary,
  ToolCallEvent,
} from "./types";
import { AGENT_NAMES } from "./types";

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

function makeAgents() {
  return Object.fromEntries(
    AGENT_NAMES.map((a) => [a, { status: "idle" as const, toolCalls: [] }])
  ) as unknown as AppState["agents"];
}

const INITIAL: AppState = {
  status: "idle",
  threadId: null,
  agents: makeAgents(),
  finalReport: "",
  citations: [],
  qualityScore: 0,
  isApproved: false,
  keyFindings: [],
  evidenceQuality: "",
  evidenceGrade: "",
  tokenSummary: null,
  rejectionMessage: null,
  errorMessage: null,
  pipelineErrors: [],
};

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function reducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "RESET":
      return { ...INITIAL, agents: makeAgents() };

    case "START":
      return { ...INITIAL, agents: makeAgents(), status: "running", threadId: action.threadId };

    case "UPDATE": {
      const ev = action.data;
      const agents = { ...state.agents };

      if (ev.type === "coordinator_decision") {
        agents.coordinator = {
          ...agents.coordinator,
          status: "complete",
          data: {
            phase: ev.phase ?? "",
            action: ev.action ?? "",
            out_of_scope: ev.out_of_scope ?? false,
            scope_rejection_reason: ev.scope_rejection_reason ?? "",
            clarification_needed: ev.clarification_needed ?? false,
            clarification_question: ev.clarification_question ?? "",
          },
        };
        // Mark the next agent as running
        const next = ev.action as AgentName | undefined;
        if (next && AGENT_NAMES.includes(next) && next !== "coordinator") {
          agents[next] = { ...agents[next], status: "running" };
        }
      } else if (ev.type === "agent_complete" && ev.agent) {
        const agentName = ev.agent as AgentName;
        agents[agentName] = {
          ...agents[agentName],
          status: "complete",
          data: ev as unknown as AppState["agents"][AgentName]["data"],
        };
        // Coordinator is deciding next
        agents.coordinator = { ...agents.coordinator, status: "running" };
      }

      return {
        ...state,
        agents,
        tokenSummary: (ev.token_summary as TokenSummary) ?? state.tokenSummary,
      };
    }

    case "TOOL_CALL": {
      const call = action.data as ToolCallEvent;
      const agentName = call.agent as AgentName;
      if (!AGENT_NAMES.includes(agentName)) return state;

      const agents = { ...state.agents };
      const existing = agents[agentName].toolCalls;

      if (call.phase === "start") {
        agents[agentName] = {
          ...agents[agentName],
          toolCalls: [...existing, call],
        };
      } else {
        // Update the matching call with output
        agents[agentName] = {
          ...agents[agentName],
          toolCalls: existing.map((tc) =>
            tc.run_id === call.run_id
              ? { ...tc, output: call.output, phase: "end" as const }
              : tc
          ),
        };
      }

      return { ...state, agents };
    }

    case "COMPLETE": {
      const d = action.data as CompletionData;
      let status: AppState["status"] = "complete";
      let rejectionMessage: string | null = null;

      if (d.out_of_scope) {
        status = "rejected";
        rejectionMessage = d.scope_rejection_reason || "Request is outside the medical evidence domain.";
      } else if (d.clarification_needed) {
        status = "rejected";
        rejectionMessage = `Clarification needed: ${d.clarification_question}`;
      } else if (d.surface_error) {
        status = "error";
      }

      return {
        ...state,
        status,
        finalReport: d.draft_report ?? "",
        citations: d.citations ?? [],
        qualityScore: d.quality_score ?? 0,
        isApproved: d.is_approved ?? false,
        keyFindings: d.key_findings ?? [],
        evidenceQuality: d.evidence_quality ?? "",
        evidenceGrade: d.evidence_grade ?? "",
        tokenSummary: d.token_summary ?? state.tokenSummary,
        rejectionMessage,
        errorMessage: d.surface_error ? (d.pipeline_error_message || "Pipeline error") : null,
        pipelineErrors: d.errors ?? [],
      };
    }

    case "ERROR":
      return { ...state, status: "error", errorMessage: action.message };

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [state, dispatch] = useReducer(reducer, INITIAL);

  // SSE subscription
  useEffect(() => {
    if (!state.threadId) return;

    const es = openStream(state.threadId);

    es.addEventListener("update", (e: Event) => {
      const data = JSON.parse((e as MessageEvent).data) as UpdateEvent;
      dispatch({ type: "UPDATE", data });
    });

    es.addEventListener("tool_call", (e: Event) => {
      const data = JSON.parse((e as MessageEvent).data) as ToolCallEvent;
      dispatch({ type: "TOOL_CALL", data });
    });

    es.addEventListener("pipeline_complete", (e: Event) => {
      const data = JSON.parse((e as MessageEvent).data) as CompletionData;
      dispatch({ type: "COMPLETE", data });
      es.close();
    });

    es.addEventListener("pipeline_error", (e: Event) => {
      const data = JSON.parse((e as MessageEvent).data) as { message: string };
      dispatch({ type: "ERROR", message: data.message });
      es.close();
    });

    es.addEventListener("stream_end", () => es.close());
    es.onerror = () => {
      if (state.status === "running") {
        dispatch({ type: "ERROR", message: "Connection to server lost." });
      }
      es.close();
    };

    return () => es.close();
  }, [state.threadId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = async (question: string) => {
    const threadId = await startRun({ question });
    dispatch({ type: "START", threadId });
  };

  const isRunning = state.status === "running";
  const isDone = state.status !== "idle" && state.status !== "running";

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <Activity className="w-6 h-6 text-blue-600" />
          <div>
            <h1 className="font-bold text-gray-900 text-lg leading-tight">
              Medical Evidence Synthesis
            </h1>
            <p className="text-xs text-gray-500">Powered by LangGraph + Gemini</p>
          </div>
          {isDone && (
            <button
              onClick={() => dispatch({ type: "RESET" })}
              className="ml-auto text-sm text-blue-600 hover:underline"
            >
              New query
            </button>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Query input */}
        <QueryInput onSubmit={handleSubmit} disabled={isRunning} />

        {/* Rejection / clarification banner */}
        {state.status === "rejected" && state.rejectionMessage && (
          <div className="flex items-start gap-3 bg-yellow-50 border border-yellow-200 rounded-xl p-4">
            <MessageSquareWarning className="w-5 h-5 text-yellow-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-yellow-800 text-sm">Pipeline halted</p>
              <p className="text-yellow-700 text-sm mt-0.5">{state.rejectionMessage}</p>
            </div>
          </div>
        )}

        {/* Error banner */}
        {state.status === "error" && state.errorMessage && (
          <div className="flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl p-4">
            <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-red-800 text-sm">Pipeline error</p>
              <p className="text-red-700 text-sm mt-0.5">{state.errorMessage}</p>
            </div>
          </div>
        )}

        {/* Pipeline in progress or done */}
        {state.status !== "idle" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left: agent progress + report */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
                <h2 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-500" />
                  Pipeline Progress
                </h2>
                <div className="space-y-2">
                  {AGENT_NAMES.map((agent) => (
                    <AgentCard key={agent} agent={agent} state={state.agents[agent]} />
                  ))}
                </div>
              </div>

              {state.finalReport && (
                <FinalReport report={state.finalReport} citations={state.citations} />
              )}
            </div>

            {/* Right: metrics */}
            <div className="space-y-4">
              <MetricsPanel
                tokenSummary={state.tokenSummary}
                qualityScore={state.qualityScore}
                isApproved={state.isApproved}
                evidenceGrade={state.evidenceGrade}
              />

              {/* Non-fatal errors */}
              {state.pipelineErrors.length > 0 && (
                <div className="bg-white rounded-xl border border-orange-200 p-4">
                  <p className="text-xs font-semibold text-orange-700 mb-2">
                    Non-fatal warnings ({state.pipelineErrors.length})
                  </p>
                  <ul className="space-y-1">
                    {state.pipelineErrors.map((e, i) => (
                      <li key={i} className="text-xs text-orange-600">
                        • {e}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
