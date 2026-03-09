import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Copy, Download, Check, BookOpen } from "lucide-react";

interface Props {
  report: string;
  citations: string[];
}

export default function FinalReport({ report, citations }: Props) {
  const [copied, setCopied] = useState(false);

  if (!report) return null;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clinical-evidence-report.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-2 px-6 py-4 border-b border-gray-100">
        <FileText className="w-5 h-5 text-blue-600" />
        <h2 className="font-semibold text-gray-800 flex-1">Final Report</h2>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 bg-gray-50 hover:bg-gray-100 rounded-lg px-3 py-1.5 transition-colors"
        >
          {copied ? (
            <><Check className="w-3.5 h-3.5 text-green-500" /> Copied</>
          ) : (
            <><Copy className="w-3.5 h-3.5" /> Copy</>
          )}
        </button>
        <button
          onClick={handleDownload}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 bg-gray-50 hover:bg-gray-100 rounded-lg px-3 py-1.5 transition-colors"
        >
          <Download className="w-3.5 h-3.5" /> Download
        </button>
      </div>

      {/* Report body */}
      <div className="px-6 py-5 prose max-w-none overflow-x-auto">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
      </div>

      {/* Citations */}
      {citations.length > 0 && (
        <div className="border-t border-gray-100 px-6 py-4">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-4 h-4 text-gray-500" />
            <h3 className="font-semibold text-sm text-gray-700">
              References ({citations.length})
            </h3>
          </div>
          <ol className="space-y-1">
            {citations.map((c, i) => (
              <li key={i} className="text-xs text-gray-600 flex gap-2">
                <span className="text-gray-400 shrink-0">{i + 1}.</span>
                <span>{c}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
