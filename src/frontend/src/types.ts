export type AgentStatus = "idle" | "running" | "complete" | "error";

export type AppStatus = "idle" | "running" | "complete" | "rejected" | "error";

// ---- Token tracking --------------------------------------------------------

export interface AgentTokens {
  input: number;
  output: number;
  calls: number;
}

export interface TokenSummary {
  by_agent: Record<string, AgentTokens>;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

// ---- Per-agent data --------------------------------------------------------

export interface ResearchData {
  research_queries: string[];
  source_count: number;
  search_summary: {
    total_sources?: number;
    by_study_type?: Record<string, number>;
    by_quality?: { high?: number; medium?: number; low?: number };
    summary_message?: string;
  };
  research_summary_excerpt: string;
  errors: string[];
}

export interface AnalysisData {
  key_findings: string[];
  evidence_quality: string;
  evidence_grade: string;
  bias_assessment: string;
  statistical_summary: Record<string, unknown>;
  errors: string[];
}

export interface WritingData {
  report_chars: number;
  citation_count: number;
  errors: string[];
}

export interface QualityData {
  quality_score: number;
  is_approved: boolean;
  quality_feedback: string[];
  errors: string[];
}

export interface CoordinatorData {
  phase: string;
  action: string;
  out_of_scope: boolean;
  scope_rejection_reason: string;
  clarification_needed: boolean;
  clarification_question: string;
}

export type AgentData =
  | ({ agent: "research" } & ResearchData)
  | ({ agent: "analysis" } & AnalysisData)
  | ({ agent: "writing" } & WritingData)
  | ({ agent: "quality" } & QualityData);

// ---- Tool calls ------------------------------------------------------------

export interface ToolCallEvent {
  agent: string;
  tool: string;
  input: string;
  output?: string;
  phase: "start" | "end";
  run_id: string;
}

export interface AgentState {
  status: AgentStatus;
  data?: AgentData | CoordinatorData;
  toolCalls: ToolCallEvent[];
}

// ---- SSE event payloads ----------------------------------------------------

export interface UpdateEvent {
  type: "coordinator_decision" | "agent_complete" | "node_update";
  node: string;
  agent?: string;
  token_summary?: TokenSummary;
  // coordinator_decision fields
  phase?: string;
  action?: string;
  out_of_scope?: boolean;
  scope_rejection_reason?: string;
  clarification_needed?: boolean;
  clarification_question?: string;
  // agent_complete fields
  [key: string]: unknown;
}

export interface CompletionData {
  draft_report: string;
  citations: string[];
  quality_score: number;
  is_approved: boolean;
  key_findings: string[];
  evidence_quality: string;
  evidence_grade: string;
  out_of_scope: boolean;
  scope_rejection_reason: string;
  clarification_needed: boolean;
  clarification_question: string;
  surface_error: boolean;
  pipeline_error_message: string;
  errors: string[];
  token_summary: TokenSummary;
}

// ---- Application state ------------------------------------------------------

export const AGENT_NAMES = ["coordinator", "research", "analysis", "writing", "quality"] as const;
export type AgentName = (typeof AGENT_NAMES)[number];

export interface AppState {
  status: AppStatus;
  threadId: string | null;
  agents: Record<AgentName, AgentState>;
  finalReport: string;
  citations: string[];
  qualityScore: number;
  isApproved: boolean;
  keyFindings: string[];
  evidenceQuality: string;
  evidenceGrade: string;
  tokenSummary: TokenSummary | null;
  rejectionMessage: string | null;
  errorMessage: string | null;
  pipelineErrors: string[];
}

export type AppAction =
  | { type: "START"; threadId: string }
  | { type: "UPDATE"; data: UpdateEvent }
  | { type: "TOOL_CALL"; data: ToolCallEvent }
  | { type: "COMPLETE"; data: CompletionData }
  | { type: "ERROR"; message: string }
  | { type: "RESET" };
