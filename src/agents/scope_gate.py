"""Scope gate utilities — keyword signal sets used by the Coordinator.

This module is a pure utility layer; it contains no LangGraph nodes or
routers.  All routing logic lives in the Coordinator.
"""

# ---------------------------------------------------------------------------
# Keyword signal sets
# ---------------------------------------------------------------------------

# Any match → immediately classified as in-scope (medical/health domain)
MEDICAL_SIGNALS: frozenset[str] = frozenset(
    {
        "clinical", "trial", "rct", "randomised", "randomized", "placebo",
        "controlled", "systematic review", "meta-analysis", "meta analysis",
        "cochrane", "pubmed", "clinicaltrials",
        "telemedicine", "telehealth", "remote monitoring",
        "diabetes", "insulin", "hba1c", "glycaemic", "glycemic",
        "therapy", "treatment", "intervention", "drug", "medication", "dose",
        "dosage", "pharmacology", "pharmacokinetics",
        "patient", "cohort", "incidence", "prevalence", "mortality", "morbidity",
        "disease", "disorder", "syndrome", "symptoms", "diagnosis", "prognosis",
        "pathology", "aetiology", "etiology",
        "cancer", "oncology", "tumour", "tumor",
        "cardiovascular", "cardiac", "hypertension", "blood pressure",
        "neurology", "psychiatric", "mental health",
        "surgery", "surgical", "procedure",
        "hospital", "physician", "nurse", "clinician", "healthcare",
        "epidemiology", "public health",
        "biomarker", "efficacy", "evidence-based", "evidence based",
        "medical", "health outcome",
    }
)

# Any match AND no medical keyword → immediately out-of-scope (no LLM cost)
HARD_OOT_SIGNALS: frozenset[str] = frozenset(
    {
        "recipe", "cooking", "cuisine", "ingredient", "baking",
        "sports score", "football score", "cricket score",
        "weather forecast", "temperature tomorrow",
        "stock price", "cryptocurrency", "bitcoin", "trading",
        "travel itinerary", "hotel booking", "flight booking",
        "restaurant review", "food review",
        "movie review", "film review", "tv show", "celebrity gossip",
        "fashion", "clothing", "makeup tutorial",
        "video game", "esports",
        "horoscope", "astrology", "zodiac",
        "relationship advice", "dating tips",
        "home renovation", "interior design",
        "car review", "automotive review",
    }
)


def keyword_scope_check(request: str) -> str | None:
    """
    Fast keyword pre-filter for domain relevance.

    Returns:
        "in_scope"      if a medical/health keyword is found
        "out_of_scope"  if only hard off-topic keywords are found
        None            if the request is ambiguous (LLM needed)
    """
    lower = request.lower()
    if any(kw in lower for kw in MEDICAL_SIGNALS):
        return "in_scope"
    if any(kw in lower for kw in HARD_OOT_SIGNALS):
        return "out_of_scope"
    return None
