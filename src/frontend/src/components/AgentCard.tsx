import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  CheckCircle2,
  XCircle,
  Circle,
  Brain,
  Search,
  BarChart2,
  FileText,
  ShieldCheck,
  Wrench,
} from "lucide-react";
import type { AgentName, AgentState, AgentStatus, ToolCallEvent } from "../types";

// ---- helpers ---------------------------------------------------------------

type AnyData = Record<string, unknown>;

const AGENT_META: Record<AgentName, { label: string; Icon: React.FC<{ className?: string }> }> = {
  coordinator: { label: "Coordinator", Icon: Brain },
  research: { label: "Research", Icon: Search },
  analysis: { label: "Analysis", Icon: BarChart2 },
  writing: { label: "Writing", Icon: FileText },
  quality: { label: "Quality Review", Icon: ShieldCheck },
};

function StatusIcon({ status }: { status: AgentStatus }) {
  switch (status) {
    case "running":
      return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
    case "complete":
      return <CheckCircle2 className="w-4 h-4 text-green-500" />;
    case "error":
      return <XCircle className="w-4 h-4 text-red-500" />;
    default:
      return <Circle className="w-4 h-4 text-gray-300" />;
  }
}

function statusRingClass(status: AgentStatus) {
  switch (status) {
    case "running":  return "bg-blue-50 text-blue-700 ring-blue-200";
    case "complete": return "bg-green-50 text-green-700 ring-green-200";
    case "error":    return "bg-red-50 text-red-700 ring-red-200";
    default:         return "bg-gray-50 text-gray-500 ring-gray-200";
  }
}

function statusLabel(status: AgentStatus) {
  switch (status) {
    case "running":  return "Running";
    case "complete": return "Complete";
    case "error":    return "Error";
    default:         return "Waiting";
  }
}

function str(v: unknown): string {
  return typeof v === "string" ? v : String(v ?? "");
}

function arr<T>(v: unknown): T[] {
  return Array.isArray(v) ? (v as T[]) : [];
}

function obj(v: unknown): AnyData {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as AnyData) : {};
}

// ---- Tool calls section ----------------------------------------------------

function ToolCallRow({ call }: { call: ToolCallEvent }) {
  const [open, setOpen] = useState(false);
  const isRunning = call.phase === "start";

  return (
    <div className="border border-gray-100 rounded">
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-gray-50 transition-colors"
        onClick={() => setOpen((v) => !v)}
      >
        <Wrench className="w-3 h-3 text-gray-400 shrink-0" />
        <span className="flex-1 text-xs font-mono text-gray-700 truncate">{call.tool}</span>
        {isRunning ? (
          <Loader2 className="w-3 h-3 text-blue-400 animate-spin shrink-0" />
        ) : (
          <CheckCircle2 className="w-3 h-3 text-green-400 shrink-0" />
        )}
        <ChevronDown
          className={`w-3 h-3 text-gray-400 shrink-0 transition-transform ${open ? "" : "-rotate-90"}`}
        />
      </button>

      {open && (
        <div className="px-2 pb-2 space-y-1.5">
          <div>
            <p className="text-xs font-medium text-gray-400 mb-0.5">Input</p>
            <pre className="text-xs text-gray-600 bg-gray-50 rounded p-1.5 whitespace-pre-wrap break-all overflow-auto max-h-40">
              {call.input}
            </pre>
          </div>
          {call.output != null && (
            <div>
              <p className="text-xs font-medium text-gray-400 mb-0.5">Output</p>
              <pre className="text-xs text-gray-600 bg-gray-50 rounded p-1.5 whitespace-pre-wrap break-all overflow-auto max-h-48">
                {call.output}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ToolCallsSection({ toolCalls }: { toolCalls: ToolCallEvent[] }) {
  if (toolCalls.length === 0) return null;
  return (
    <div className="mt-3">
      <p className="font-medium text-gray-500 mb-1 text-xs flex items-center gap-1">
        <Wrench className="w-3 h-3" /> Tool calls ({toolCalls.length})
      </p>
      <div className="space-y-1">
        {toolCalls.map((tc) => (
          <ToolCallRow key={tc.run_id} call={tc} />
        ))}
      </div>
    </div>
  );
}

// ---- Sub-content -----------------------------------------------------------

function CoordinatorContent({ data }: { data: AnyData }) {
  const phase = str(data.phase);
  const action = str(data.action);
  return (
    <div className="text-sm space-y-1 text-gray-700">
      {phase && (
        <div>
          <span className="font-medium text-gray-500">Phase: </span>
          <span className="font-mono text-xs bg-gray-100 rounded px-1">{phase}</span>
        </div>
      )}
      {action && (
        <div>
          <span className="font-medium text-gray-500">Next: </span>
          <span className="font-mono text-xs bg-blue-50 text-blue-700 rounded px-1">{action}</span>
        </div>
      )}
      {!!data.clarification_needed && (
        <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-yellow-800 text-xs">
          <strong>Clarification needed:</strong> {str(data.clarification_question)}
        </div>
      )}
      {!!data.out_of_scope && (
        <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-red-800 text-xs">
          <strong>Out of scope:</strong> {str(data.scope_rejection_reason)}
        </div>
      )}
    </div>
  );
}

function ResearchContent({ data }: { data: AnyData }) {
  const queries = arr<string>(data.research_queries);
  const searchSummary = obj(data.search_summary);
  const excerpt = str(data.research_summary_excerpt);
  const byQuality = obj(searchSummary.by_quality);

  return (
    <div className="text-sm space-y-3 text-gray-700">
      {!!searchSummary.summary_message && (
        <p className="text-blue-700 font-medium text-xs">{str(searchSummary.summary_message)}</p>
      )}

      {Object.keys(byQuality).length > 0 && (
        <div className="flex gap-3 text-xs">
          {(["high", "medium", "low"] as const).map((tier) => {
            const count = (byQuality[tier] as number) ?? 0;
            const cls =
              tier === "high"
                ? "bg-green-100 text-green-700"
                : tier === "medium"
                ? "bg-yellow-100 text-yellow-700"
                : "bg-gray-100 text-gray-600";
            return (
              <span key={tier} className={`${cls} rounded-full px-2 py-0.5 font-medium`}>
                {count} {tier}
              </span>
            );
          })}
        </div>
      )}

      {queries.length > 0 && (
        <div>
          <p className="font-medium text-gray-500 mb-1">Queries run:</p>
          <ul className="space-y-0.5">
            {queries.map((q, i) => (
              <li key={i} className="text-xs text-gray-600 flex gap-1">
                <span className="text-gray-400 select-none">{i + 1}.</span> {q}
              </li>
            ))}
          </ul>
        </div>
      )}

      {excerpt && (
        <div>
          <p className="font-medium text-gray-500 mb-1">Research summary:</p>
          <p className="text-xs text-gray-600 leading-relaxed">{excerpt}</p>
        </div>
      )}
    </div>
  );
}

function AnalysisContent({ data }: { data: AnyData }) {
  const findings = arr<string>(data.key_findings);
  const quality = str(data.evidence_quality);
  const grade = str(data.evidence_grade);
  const bias = str(data.bias_assessment);

  const qualityColor =
    quality === "strong"
      ? "text-green-700 bg-green-50"
      : quality === "moderate"
      ? "text-yellow-700 bg-yellow-50"
      : "text-red-700 bg-red-50";

  return (
    <div className="text-sm space-y-3 text-gray-700">
      <div className="flex gap-2 flex-wrap text-xs font-medium">
        {quality && (
          <span className={`rounded-full px-2 py-0.5 ${qualityColor}`}>
            Evidence: {quality}
          </span>
        )}
        {grade && (
          <span className="rounded-full px-2 py-0.5 bg-blue-50 text-blue-700">
            GRADE: {grade}
          </span>
        )}
        {bias && (
          <span className="rounded-full px-2 py-0.5 bg-purple-50 text-purple-700">
            Bias: {bias}
          </span>
        )}
      </div>

      {findings.length > 0 && (
        <div>
          <p className="font-medium text-gray-500 mb-1">Key findings ({findings.length}):</p>
          <ul className="space-y-1">
            {findings.map((f, i) => (
              <li key={i} className="text-xs text-gray-600 flex gap-1.5">
                <span className="text-blue-400 mt-0.5 shrink-0">•</span> {f}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function WritingContent({ data }: { data: AnyData }) {
  const chars = (data.report_chars as number) ?? 0;
  const citations = (data.citation_count as number) ?? 0;
  return (
    <div className="text-sm space-y-1 text-gray-700">
      <div className="flex gap-4 text-xs">
        <span className="bg-blue-50 text-blue-700 rounded px-2 py-0.5">
          {chars ? `~${Math.round(chars / 5)} words` : "—"}
        </span>
        <span className="bg-gray-100 text-gray-700 rounded px-2 py-0.5">
          {citations} citation{citations !== 1 ? "s" : ""}
        </span>
      </div>
    </div>
  );
}

function QualityContent({ data }: { data: AnyData }) {
  const score = (data.quality_score as number) ?? 0;
  const approved = !!data.is_approved;
  const feedback = arr<string>(data.quality_feedback);

  const scoreColor =
    score >= 0.8 ? "text-green-700" : score >= 0.6 ? "text-yellow-700" : "text-red-700";
  const scoreBg =
    score >= 0.8 ? "bg-green-50" : score >= 0.6 ? "bg-yellow-50" : "bg-red-50";

  return (
    <div className="text-sm space-y-3 text-gray-700">
      <div className="flex items-center gap-3">
        <div className={`${scoreBg} ${scoreColor} rounded-lg px-3 py-1.5 text-center`}>
          <div className="text-2xl font-bold">{(score * 100).toFixed(0)}</div>
          <div className="text-xs font-medium">/ 100</div>
        </div>
        <span
          className={`text-xs font-semibold rounded-full px-3 py-1 ${
            approved ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
          }`}
        >
          {approved ? "Approved" : "Revision needed"}
        </span>
      </div>

      {feedback.length > 0 && (
        <div>
          <p className="font-medium text-gray-500 mb-1 text-xs">Feedback:</p>
          <ul className="space-y-1">
            {feedback.map((f, i) => (
              <li key={i} className="text-xs text-gray-600 flex gap-1.5">
                <span className="text-yellow-500 shrink-0">›</span> {f}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function AgentContent({ agent, state }: { agent: AgentName; state: AgentState }) {
  const d = state.data as unknown as AnyData | undefined;
  if (!d || state.status === "idle") return null;

  if (agent === "coordinator") return <CoordinatorContent data={d} />;
  if (agent === "research")    return <ResearchContent data={d} />;
  if (agent === "analysis")    return <AnalysisContent data={d} />;
  if (agent === "writing")     return <WritingContent data={d} />;
  if (agent === "quality")     return <QualityContent data={d} />;
  return null;
}

// ---- Main card -------------------------------------------------------------

interface AgentCardProps {
  agent: AgentName;
  state: AgentState;
}

export default function AgentCard({ agent, state }: AgentCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { label, Icon } = AGENT_META[agent];
  const hasContent = state.status !== "idle" && (state.data != null || state.toolCalls.length > 0);
  const errors = arr<string>((state.data as unknown as AnyData | undefined)?.errors);

  const borderClass =
    state.status === "running"
      ? "border-blue-300 shadow-blue-100 shadow-md"
      : state.status === "complete"
      ? "border-green-200"
      : state.status === "error"
      ? "border-red-200"
      : "border-gray-200";

  return (
    <div className={`bg-white rounded-lg border transition-all ${borderClass}`}>
      <button
        className="w-full flex items-center gap-3 px-4 py-3 text-left"
        onClick={() => hasContent && setExpanded((v) => !v)}
        disabled={!hasContent}
      >
        <Icon className="w-4 h-4 text-gray-500 shrink-0" />
        <span className="flex-1 text-sm font-medium text-gray-800">{label}</span>

        {state.toolCalls.length > 0 && (
          <span className="text-xs text-gray-400 flex items-center gap-0.5">
            <Wrench className="w-3 h-3" /> {state.toolCalls.length}
          </span>
        )}

        <span className={`text-xs font-medium ring-1 rounded-full px-2 py-0.5 ${statusRingClass(state.status)}`}>
          {statusLabel(state.status)}
        </span>

        <StatusIcon status={state.status} />

        {hasContent && (
          expanded
            ? <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
            : <ChevronRight className="w-3.5 h-3.5 text-gray-400" />
        )}
      </button>

      {hasContent && expanded && (
        <div className="px-4 pb-4 border-t border-gray-100 pt-3">
          <AgentContent agent={agent} state={state} />

          <ToolCallsSection toolCalls={state.toolCalls} />

          {errors.length > 0 && (
            <div className="mt-2 p-2 bg-red-50 rounded text-xs text-red-700">
              <strong>Errors:</strong> {errors.join("; ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
