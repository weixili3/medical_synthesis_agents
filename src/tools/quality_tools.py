"""Quality assurance and validation tools for the Quality Agent."""

import json
import logging
import re

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Readability Assessment
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    """Count syllables in a word using a vowel-group heuristic."""
    word = word.lower().rstrip(".,;:!?\"'")
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent trailing 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _python_readability(text: str) -> dict:
    """Compute readability metrics in pure Python (no external dependencies)."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    n_words = max(len(words), 1)
    syllables = [_count_syllables(w) for w in words]
    n_syllables = sum(syllables)
    n_complex = sum(1 for s in syllables if s >= 3)
    n_chars = sum(len(w) for w in words)

    asl = n_words / n_sentences      # avg sentence length
    asw = n_syllables / n_words       # avg syllables per word

    flesch_ease = round(206.835 - 1.015 * asl - 84.6 * asw, 1)
    fk_grade = round(0.39 * asl + 11.8 * asw - 15.59, 1)
    gunning_fog = round(0.4 * (asl + 100 * n_complex / n_words), 1)
    coleman_liau = round(5.879851 * (n_chars / n_words) - 29.587280 * (n_sentences / n_words) - 15.800804, 1)
    smog = round(3 + (30 * n_complex / n_sentences) ** 0.5, 1) if n_sentences >= 30 else round(3 + n_complex ** 0.5, 1)
    ari = round(4.71 * (n_chars / n_words) + 0.5 * asl - 21.43, 1)

    return {
        "flesch_reading_ease": flesch_ease,
        "flesch_kincaid_grade": fk_grade,
        "gunning_fog_index": gunning_fog,
        "coleman_liau_index": coleman_liau,
        "smog_index": smog,
        "automated_readability_index": ari,
        "word_count": n_words,
        "sentence_count": n_sentences,
    }


@tool
def check_readability(text: str) -> str:
    """
    Assess the readability of a text document using multiple standard metrics.

    Computes Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog
    Index, Coleman-Liau Index, and SMOG Index.

    Args:
        text: The text to analyse.

    Returns:
        JSON string with readability scores and an overall interpretation.
    """
    if len(text.strip()) < 100:
        return json.dumps({"error": "Text is too short for reliable readability scoring (minimum 100 chars)."})

    metrics: dict = {}
    note = ""

    try:
        import textstat
        metrics = {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 1),
            "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 1),
            "gunning_fog_index": round(textstat.gunning_fog(text), 1),
            "coleman_liau_index": round(textstat.coleman_liau_index(text), 1),
            "smog_index": round(textstat.smog_index(text), 1),
            "automated_readability_index": round(textstat.automated_readability_index(text), 1),
            "word_count": textstat.lexicon_count(text),
            "sentence_count": textstat.sentence_count(text),
        }
    except Exception as exc:
        logger.warning("textstat unavailable (%s), using pure-Python fallback.", exc)
        metrics = _python_readability(text)
        note = "Pure-Python metrics (textstat unavailable)"

    # Interpret Flesch Reading Ease (0–100, higher = easier)
    fe = metrics["flesch_reading_ease"]
    if fe >= 70:
        interpretation = "Easy to read (general public)"
    elif fe >= 50:
        interpretation = "Fairly difficult (high school / undergraduate level)"
    elif fe >= 30:
        interpretation = "Difficult (college / professional level)"
    else:
        interpretation = "Very difficult (expert / academic level)"

    result = {**metrics, "interpretation": interpretation}
    if note:
        result["note"] = note
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Completeness Check
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "executive summary",
    "introduction",
    "methodology",
    "key findings",
    "evidence analysis",
    "discussion",
    "conclusions",
    "references",
]

CLINICAL_REQUIRED = [
    "executive summary",
    "introduction",
    "methodology",
    "key findings",
    "evidence analysis",
    "discussion",
    "conclusions",
    "limitations",
    "references",
]


@tool
def check_completeness(report_text: str, content_type: str = "general") -> str:
    """
    Verify that a report contains all required sections for its content type.

    Args:
        report_text: The full text of the report (markdown format preferred).
        content_type: 'clinical' for clinical/evidence reports, 'general' otherwise.

    Returns:
        JSON string with keys: missing_sections, present_sections,
        completeness_score (0–1), issues.
    """
    required = CLINICAL_REQUIRED if content_type.lower() == "clinical" else REQUIRED_SECTIONS

    report_lower = report_text.lower()

    present = []
    missing = []
    for section in required:
        # Check for heading patterns (# Section or **Section**)
        pattern = rf"(?:^|\n)#{1,4}\s+{re.escape(section)}|(?:\*\*{re.escape(section)}\*\*)"
        if re.search(pattern, report_lower, re.IGNORECASE) or section in report_lower:
            present.append(section)
        else:
            missing.append(section)

    completeness_score = len(present) / len(required) if required else 1.0

    issues = []
    if missing:
        issues.append(f"Missing sections: {', '.join(missing)}")
    if len(report_text.split()) < 300:
        issues.append("Report appears too short (fewer than 300 words).")
    if "references" in missing or "citations" not in report_lower:
        issues.append("No references/citations section detected.")

    result = {
        "required_sections": required,
        "present_sections": present,
        "missing_sections": missing,
        "completeness_score": round(completeness_score, 2),
        "word_count": len(report_text.split()),
        "issues": issues,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Grammar & Style Checking
# ---------------------------------------------------------------------------


@tool
def check_grammar(text: str, max_errors: int = 20) -> str:
    """
    Check the text for grammar and style issues using LanguageTool.

    Falls back to a simple rule-based check if LanguageTool is unavailable.

    Args:
        text: The text to check.
        max_errors: Maximum number of errors to report (default 20).

    Returns:
        JSON string with error count, sample issues, and a quality label.
    """
    try:
        import language_tool_python

        tool_instance = language_tool_python.LanguageTool("en-US")
        matches = tool_instance.check(text)[:max_errors]

        issues = [
            {
                "message": m.message,
                "category": m.category,
                "context": m.context.strip(),
                "suggestion": m.replacements[:3] if m.replacements else [],
            }
            for m in matches
        ]

        error_count = len(matches)
        if error_count == 0:
            quality = "excellent"
        elif error_count <= 5:
            quality = "good"
        elif error_count <= 15:
            quality = "fair"
        else:
            quality = "needs improvement"

        return json.dumps(
            {"error_count": error_count, "quality": quality, "issues": issues},
            indent=2,
        )

    except ImportError:
        pass
    except Exception as exc:
        logger.warning("LanguageTool check failed, using fallback: %s", exc)

    # Fallback: simple heuristic checks
    return _simple_grammar_check(text, max_errors)


def _simple_grammar_check(text: str, max_errors: int) -> str:
    """Basic heuristic grammar checks when LanguageTool is unavailable."""
    issues = []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for i, sent in enumerate(sentences[:50]):
        if sent and not sent[0].isupper() and not sent.startswith("#"):
            issues.append({"message": "Sentence may not start with a capital letter.", "context": sent[:80]})
        if re.search(r"\b(\w+) \1\b", sent, re.IGNORECASE):
            issues.append({"message": "Possible duplicate word.", "context": sent[:80]})

    issues = issues[:max_errors]
    error_count = len(issues)
    quality = "good" if error_count <= 3 else "fair"

    return json.dumps(
        {"error_count": error_count, "quality": quality, "issues": issues, "note": "Heuristic check only."},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Relevancy Check
# ---------------------------------------------------------------------------


@tool
def check_relevancy(report_text: str, original_request: str) -> str:
    """
    Assess how relevant the report is to the original content request.

    Uses keyword overlap and section alignment as a proxy for relevancy.

    Args:
        report_text: The generated report text.
        original_request: The original user request string.

    Returns:
        JSON string with relevancy_score (0–1), matched_keywords, and feedback.
    """
    # Extract meaningful keywords from the request
    stopwords = {
        "a", "an", "the", "for", "of", "in", "and", "or", "to", "with",
        "is", "are", "be", "on", "at", "by", "as", "it", "its", "this",
        "that", "from", "create", "comprehensive", "provide", "give",
    }
    request_words = set(
        w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", original_request)
        if w.lower() not in stopwords
    )

    report_lower = report_text.lower()
    matched = [w for w in request_words if w in report_lower]
    relevancy_score = len(matched) / len(request_words) if request_words else 1.0

    feedback = []
    if relevancy_score < 0.5:
        feedback.append("Report may not adequately address the original request. Key topics may be missing.")
    missing_kw = [w for w in request_words if w not in report_lower]
    if missing_kw:
        feedback.append(f"Keywords from the request not found in report: {', '.join(missing_kw[:5])}")
    if relevancy_score >= 0.8:
        feedback.append("Report appears highly relevant to the original request.")

    return json.dumps(
        {
            "relevancy_score": round(relevancy_score, 2),
            "matched_keywords": matched,
            "missing_keywords": missing_kw,
            "feedback": feedback,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Medical Claim Verification
# ---------------------------------------------------------------------------


@tool
def check_medical_claims(report_text: str, sources_json: str) -> str:
    """
    Cross-reference medical claims in the report against the original source snippets.

    Extracts factual sentences from the report that contain numerical data or
    clinical assertions, then checks whether the key terms and numbers appear
    in the provided source material.

    Args:
        report_text: The full draft report text.
        sources_json: JSON string — list of source dicts, each with at minimum
                      a 'snippet' or 'full_text' field and optionally 'title'.
                      Example: '[{"title":"Smith 2021","snippet":"HbA1c reduced by 0.5%..."}]'

    Returns:
        JSON string with verified_claims, unverified_claims, accuracy_score,
        and feedback.
    """
    try:
        sources: list[dict] = json.loads(sources_json)
    except (json.JSONDecodeError, TypeError):
        sources = []

    # Aggregate all source text for cross-referencing
    source_corpus = " ".join(
        (s.get("full_text") or s.get("snippet") or "")
        for s in sources
    ).lower()

    # Extract sentences that look like factual medical claims
    # (contain a number, percentage, or key clinical terms)
    clinical_pattern = re.compile(
        r"[^.!?\n]*(?:\d+\.?\d*\s*%|\b(?:hba1c|mortality|adherence|odds ratio|relative risk|"
        r"p\s*[<=>]\s*0\.\d+|confidence interval|rct|meta-analysis|systematic review|"
        r"significant|reduced|increased|improved|associated)\b)[^.!?\n]*[.!?]",
        re.IGNORECASE,
    )
    claim_sentences = clinical_pattern.findall(report_text)

    verified: list[str] = []
    unverified: list[str] = []

    for sentence in claim_sentences[:30]:  # cap to avoid excessive processing
        sentence_stripped = sentence.strip()
        # Extract key numbers and clinical terms from the sentence
        numbers = re.findall(r"\d+\.?\d*", sentence_stripped)
        # A claim is "verified" if at least one of its numbers appears in the source corpus
        supported = any(num in source_corpus for num in numbers) if numbers else (
            sentence_stripped[:40].lower() in source_corpus
        )
        if supported:
            verified.append(sentence_stripped[:150])
        else:
            unverified.append(sentence_stripped[:150])

    total = len(verified) + len(unverified)
    accuracy_score = round(len(verified) / total, 2) if total > 0 else 1.0

    feedback = []
    if not source_corpus:
        feedback.append("No source text available for claim verification — provide source snippets.")
    if unverified:
        feedback.append(
            f"{len(unverified)} claim(s) could not be traced to the provided sources. "
            "Review for potential misinterpretation or unsupported assertions."
        )
    if accuracy_score >= 0.85:
        feedback.append("Most clinical claims appear consistent with the provided source material.")

    return json.dumps(
        {
            "verified_claims": verified[:10],
            "unverified_claims": unverified[:10],
            "verified_count": len(verified),
            "unverified_count": len(unverified),
            "accuracy_score": accuracy_score,
            "feedback": feedback,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_quality_tools() -> list:
    """Return the complete list of tools available to the Quality Agent."""
    return [check_readability, check_completeness, check_grammar, check_relevancy, check_medical_claims]
