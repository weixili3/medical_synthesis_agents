const BASE = "/api";

export interface RunOptions {
  question: string;
  max_iterations?: number;
  max_retries_per_phase?: number;
}

export async function startRun(opts: RunOptions): Promise<string> {
  const res = await fetch(`${BASE}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: opts.question,
      max_iterations: opts.max_iterations ?? 3,
      max_retries_per_phase: opts.max_retries_per_phase ?? 2,
    }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.thread_id as string;
}

export function openStream(threadId: string): EventSource {
  return new EventSource(`${BASE}/stream/${threadId}`);
}
