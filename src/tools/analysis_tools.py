"""Data analysis tools for the Analysis Agent."""

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence Analysis
# ---------------------------------------------------------------------------


@tool
def analyze_evidence(research_json: str) -> str:
    """
    Analyse a JSON array of research sources and extract structured evidence.

    Performs basic NLP-style extraction: counts sources, identifies common
    themes, classifies evidence strength, and returns a structured summary.

    Args:
        research_json: JSON string — either a list of source dicts
                       [{title, url, snippet/full_text}] or a plain string
                       summary.  Both formats are handled.

    Returns:
        JSON string with keys: source_count, evidence_strength, themes,
        key_points, methodology_types.
    """
    try:
        data = json.loads(research_json)
    except (json.JSONDecodeError, TypeError):
        # Treat as plain-text summary
        data = [{"title": "Summary", "full_text": str(research_json)}]

    sources = data if isinstance(data, list) else [data]
    source_count = len(sources)

    # Aggregate all text for theme extraction
    all_text = " ".join(
        (s.get("full_text") or s.get("snippet") or s.get("body") or "")
        for s in sources
    ).lower()

    # Simple keyword-based theme detection
    theme_keywords = {
        "randomised controlled trial": ["randomised controlled trial", "rct", "randomized controlled"],
        "systematic review": ["systematic review", "meta-analysis", "cochrane"],
        "clinical outcomes": ["clinical outcome", "patient outcome", "mortality", "morbidity"],
        "telemedicine": ["telemedicine", "telehealth", "remote monitoring", "digital health"],
        "diabetes management": ["diabetes", "glycaemic", "hba1c", "glucose", "insulin"],
        "cost-effectiveness": ["cost-effective", "economic", "cost savings", "resources"],
        "patient satisfaction": ["patient satisfaction", "patient experience", "adherence"],
    }

    detected_themes = [
        theme for theme, keywords in theme_keywords.items()
        if any(kw in all_text for kw in keywords)
    ]

    # Assess evidence strength based on study type mentions
    if any(k in all_text for k in ["randomised controlled trial", "rct", "randomized controlled"]):
        strength = "strong"
    elif any(k in all_text for k in ["systematic review", "meta-analysis", "cohort study"]):
        strength = "moderate-to-strong"
    elif any(k in all_text for k in ["observational", "case study", "survey"]):
        strength = "moderate"
    else:
        strength = "weak-to-moderate"

    methodology_types = []
    for mt in ["randomised controlled trial", "cohort study", "systematic review",
               "meta-analysis", "observational study", "case study", "survey"]:
        if mt in all_text:
            methodology_types.append(mt.title())

    result = {
        "source_count": source_count,
        "evidence_strength": strength,
        "themes": detected_themes or ["general health topic"],
        "methodology_types": methodology_types or ["not specified"],
        "key_points": [
            f"Analysed {source_count} source(s)",
            f"Evidence strength classified as: {strength}",
            f"Primary themes: {', '.join(detected_themes[:3]) if detected_themes else 'general'}",
        ],
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------


@tool
def calculate_statistics(data_json: str) -> str:
    """
    Calculate descriptive and inferential statistics on a numerical dataset.

    Args:
        data_json: JSON string of a list of numbers or a dict with a 'values' key.
                   Example: "[12.3, 14.5, 11.8]"  or  '{"values": [12.3, 14.5]}'

    Returns:
        JSON string with mean, median, std_dev, min, max, count, and
        95% confidence interval if scipy is available.
    """
    try:
        import numpy as np
        from scipy import stats as sp_stats

        raw = json.loads(data_json)
        if isinstance(raw, dict):
            values = raw.get("values", [])
        elif isinstance(raw, list):
            values = raw
        else:
            return json.dumps({"error": "Input must be a list or dict with 'values' key."})

        arr = np.array([float(v) for v in values])
        n = len(arr)

        if n == 0:
            return json.dumps({"error": "Empty dataset."})

        ci = sp_stats.t.interval(0.95, df=n - 1, loc=float(np.mean(arr)), scale=sp_stats.sem(arr)) if n > 1 else (None, None)

        result = {
            "count": n,
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std_dev": round(float(np.std(arr, ddof=1)), 4) if n > 1 else 0.0,
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "confidence_interval_95": {
                "lower": round(float(ci[0]), 4) if ci[0] is not None else None,
                "upper": round(float(ci[1]), 4) if ci[1] is not None else None,
            },
        }
        return json.dumps(result, indent=2)

    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({"error": f"Invalid data format: {exc}"})
    except Exception as exc:
        logger.warning("Statistics calculation failed: %s", exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_analysis_tools() -> list:
    """Return the complete list of tools available to the Analysis Agent."""
    return [analyze_evidence, calculate_statistics]
