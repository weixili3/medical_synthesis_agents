import { useState } from "react";
import { Search, FlaskConical } from "lucide-react";

interface Props {
  onSubmit: (question: string) => void;
  disabled: boolean;
}

const EXAMPLES = [
  "What is the clinical evidence for telemedicine in managing Type 2 diabetes?",
  "Summarise RCT evidence for GLP-1 receptor agonists in obesity treatment.",
  "What does the evidence say about cognitive behavioural therapy for depression?",
];

export default function QueryInput({ onSubmit, disabled }: Props) {
  const [question, setQuestion] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (q) onSubmit(q);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <FlaskConical className="w-5 h-5 text-blue-600" />
        <h2 className="font-semibold text-gray-800">Research Question</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a clinical evidence question, e.g. 'What is the evidence for…'"
          rows={3}
          disabled={disabled}
          className="w-full resize-none rounded-lg border border-gray-300 px-4 py-3 text-sm text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400"
        />

        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                type="button"
                disabled={disabled}
                onClick={() => setQuestion(ex)}
                className="text-xs text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-full px-3 py-1 transition-colors disabled:opacity-40"
              >
                Example {i + 1}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={disabled || !question.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
          >
            <Search className="w-4 h-4" />
            {disabled ? "Running…" : "Run Pipeline"}
          </button>
        </div>
      </form>
    </div>
  );
}
