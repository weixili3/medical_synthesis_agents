import { Coins, Zap, Award, TrendingUp } from "lucide-react";
import type { TokenSummary } from "../types";

interface Props {
  tokenSummary: TokenSummary | null;
  qualityScore: number;
  isApproved: boolean;
  evidenceGrade: string;
}

const AGENT_ORDER = ["coordinator", "research", "analysis", "writing", "quality"] as const;
const AGENT_LABEL: Record<string, string> = {
  coordinator: "Coordinator",
  research: "Research",
  analysis: "Analysis",
  writing: "Writing",
  quality: "Quality",
  unknown: "Other",
};

function fmt(n: number) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

function qualityColor(score: number) {
  if (score >= 0.8) return "text-green-600";
  if (score >= 0.6) return "text-yellow-600";
  return "text-red-600";
}

function gradeBadge(grade: string) {
  const colors: Record<string, string> = {
    A: "bg-green-100 text-green-700",
    B: "bg-blue-100 text-blue-700",
    C: "bg-yellow-100 text-yellow-700",
    D: "bg-red-100 text-red-700",
  };
  return colors[grade] ?? "bg-gray-100 text-gray-600";
}

export default function MetricsPanel({ tokenSummary, qualityScore, isApproved, evidenceGrade }: Props) {
  const hasQuality = qualityScore > 0;

  return (
    <div className="space-y-4">
      {/* Quality */}
      {hasQuality && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <div className="flex items-center gap-2 mb-3">
            <Award className="w-4 h-4 text-purple-500" />
            <h3 className="font-semibold text-sm text-gray-800">Quality</h3>
          </div>

          <div className="flex items-end gap-3">
            <span className={`text-4xl font-bold ${qualityColor(qualityScore)}`}>
              {(qualityScore * 100).toFixed(0)}
            </span>
            <span className="text-gray-400 text-sm mb-1">/ 100</span>
            <span
              className={`ml-auto text-xs font-semibold rounded-full px-2 py-1 ${
                isApproved ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
              }`}
            >
              {isApproved ? "Approved" : "Needs revision"}
            </span>
          </div>

          {evidenceGrade && (
            <div className="mt-2 flex items-center gap-2">
              <TrendingUp className="w-3.5 h-3.5 text-gray-400" />
              <span className="text-xs text-gray-500">GRADE level:</span>
              <span className={`text-xs font-bold rounded px-1.5 py-0.5 ${gradeBadge(evidenceGrade)}`}>
                {evidenceGrade}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Tokens */}
      {tokenSummary && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-yellow-500" />
            <h3 className="font-semibold text-sm text-gray-800">Token Usage</h3>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="bg-gray-50 rounded-lg p-2">
              <p className="text-xs text-gray-500">Input</p>
              <p className="text-sm font-semibold text-gray-800">
                {fmt(tokenSummary.total_input_tokens)}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-2">
              <p className="text-xs text-gray-500">Output</p>
              <p className="text-sm font-semibold text-gray-800">
                {fmt(tokenSummary.total_output_tokens)}
              </p>
            </div>
          </div>

          <div className="mb-3">
            <p className="text-xs text-gray-500 mb-1">By agent</p>
            <div className="space-y-1.5">
              {AGENT_ORDER.map((agent) => {
                const ag = tokenSummary.by_agent[agent];
                if (!ag || ag.calls === 0) return null;
                const total = ag.input + ag.output;
                const pct = tokenSummary.total_tokens
                  ? Math.round((total / tokenSummary.total_tokens) * 100)
                  : 0;
                return (
                  <div key={agent}>
                    <div className="flex justify-between text-xs text-gray-600 mb-0.5">
                      <span>{AGENT_LABEL[agent] ?? agent}</span>
                      <span>{fmt(total)} ({pct}%)</span>
                    </div>
                    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-400 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Cost */}
      {tokenSummary && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <div className="flex items-center gap-2 mb-2">
            <Coins className="w-4 h-4 text-green-500" />
            <h3 className="font-semibold text-sm text-gray-800">Estimated Cost</h3>
          </div>
          <p className="text-3xl font-bold text-gray-900">
            ${tokenSummary.estimated_cost_usd.toFixed(4)}
          </p>
          <p className="text-xs text-gray-400 mt-1">Gemini 2.0 Flash pricing</p>
        </div>
      )}
    </div>
  );
}
