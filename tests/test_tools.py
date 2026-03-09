"""Unit tests for all tool modules (search, analysis, writing, quality)."""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Search Tools
# ---------------------------------------------------------------------------


class TestGoogleSearch(unittest.TestCase):
    """Tests for the google_search tool."""

    def test_missing_credentials_returns_config_message(self):
        """Should return a helpful message when API keys are not set."""
        from src.tools.search_tools import google_search

        with patch.dict("os.environ", {}, clear=True):
            result = google_search.invoke({"query": "telemedicine diabetes"})

        self.assertIn("not configured", result.lower())

    @patch("src.tools.search_tools.requests.get")
    def test_successful_search_returns_formatted_results(self, mock_get):
        """Should return numbered results with title, URL, and snippet."""
        from src.tools.search_tools import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Telemedicine for Diabetes",
                    "link": "https://example.com/article",
                    "snippet": "A study on telemedicine interventions.",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        env = {"GOOGLE_SEARCH_API_KEY": "test_key", "GOOGLE_SEARCH_ENGINE_ID": "test_cx"}
        with patch.dict("os.environ", env):
            result = google_search.invoke({"query": "telemedicine diabetes"})

        self.assertIn("Telemedicine for Diabetes", result)
        self.assertIn("https://example.com/article", result)
        self.assertIn("[1]", result)

    @patch("src.tools.search_tools.requests.get")
    def test_empty_results_handled(self, mock_get):
        """Should return a 'no results' message when API returns no items."""
        from src.tools.search_tools import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        env = {"GOOGLE_SEARCH_API_KEY": "key", "GOOGLE_SEARCH_ENGINE_ID": "cx"}
        with patch.dict("os.environ", env):
            result = google_search.invoke({"query": "xyzzy nonexistent topic 999"})

        self.assertIn("no results", result.lower())

    @patch("src.tools.search_tools.requests.get")
    def test_http_400_error_returns_error_string(self, mock_get):
        """Should return an error string (not raise) when the API returns HTTP 400."""
        import requests as req
        from src.tools.search_tools import google_search

        mock_response = MagicMock()
        mock_response.status_code = 400
        http_error = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        env = {"GOOGLE_SEARCH_API_KEY": "bad-key", "GOOGLE_SEARCH_ENGINE_ID": "cx"}
        with patch.dict("os.environ", env):
            result = google_search.invoke({"query": "test query"})

        self.assertIn("400", result)
        self.assertIn("error", result.lower())

    @patch("src.tools.search_tools.requests.get")
    def test_http_401_error_returns_error_string(self, mock_get):
        """Should return an error string (not raise) when credentials are invalid (HTTP 401)."""
        import requests as req
        from src.tools.search_tools import google_search

        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        env = {"GOOGLE_SEARCH_API_KEY": "invalid-key", "GOOGLE_SEARCH_ENGINE_ID": "cx"}
        with patch.dict("os.environ", env):
            result = google_search.invoke({"query": "test query"})

        self.assertIn("401", result)
        self.assertIn("error", result.lower())


class TestWebScrape(unittest.TestCase):
    """Tests for the web_scrape tool."""

    @patch("src.tools.search_tools.requests.get")
    def test_scrapes_main_content(self, mock_get):
        """Should return clean text content from a webpage."""
        from src.tools.search_tools import web_scrape

        html = """
        <html><body>
          <nav>Nav junk</nav>
          <main><p>This is the main article content about telemedicine.</p></main>
          <footer>Footer junk</footer>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.content = html.encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = web_scrape.invoke({"url": "https://example.com"})

        self.assertIn("main article content", result)
        self.assertNotIn("Nav junk", result)
        self.assertNotIn("Footer junk", result)

    @patch("src.tools.search_tools.requests.get")
    def test_truncates_long_content(self, mock_get):
        """Content longer than max_chars should be truncated."""
        from src.tools.search_tools import web_scrape

        long_text = "word " * 1000
        html = f"<html><body><main><p>{long_text}</p></main></body></html>"
        mock_response = MagicMock()
        mock_response.content = html.encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = web_scrape.invoke({"url": "https://example.com", "max_chars": 100})

        self.assertLessEqual(len(result), 200)  # truncated + " [truncated]"
        self.assertIn("[truncated]", result)

    @patch("src.tools.search_tools.requests.get")
    def test_timeout_error_handled(self, mock_get):
        """Should return a descriptive message on timeout."""
        import requests as req
        from src.tools.search_tools import web_scrape

        mock_get.side_effect = req.exceptions.Timeout()
        result = web_scrape.invoke({"url": "https://slow-site.example.com"})

        self.assertIn("timeout", result.lower())


# ---------------------------------------------------------------------------
# Analysis Tools
# ---------------------------------------------------------------------------


class TestCalculateStatistics(unittest.TestCase):
    """Tests for the calculate_statistics tool."""

    def test_basic_statistics(self):
        """Should compute correct mean, median, min, max for a list of numbers."""
        from src.tools.analysis_tools import calculate_statistics

        result = calculate_statistics.invoke({"data_json": "[10, 20, 30, 40, 50]"})
        data = json.loads(result)

        self.assertEqual(data["count"], 5)
        self.assertAlmostEqual(data["mean"], 30.0)
        self.assertAlmostEqual(data["median"], 30.0)
        self.assertEqual(data["min"], 10.0)
        self.assertEqual(data["max"], 50.0)

    def test_dict_input_format(self):
        """Should accept {'values': [...]} input format."""
        from src.tools.analysis_tools import calculate_statistics

        result = calculate_statistics.invoke({"data_json": '{"values": [1.5, 2.5, 3.5]}'})
        data = json.loads(result)

        self.assertEqual(data["count"], 3)
        self.assertAlmostEqual(data["mean"], 2.5)

    def test_empty_list_returns_error(self):
        """Should return an error message for empty datasets."""
        from src.tools.analysis_tools import calculate_statistics

        result = calculate_statistics.invoke({"data_json": "[]"})
        data = json.loads(result)

        self.assertIn("error", data)

    def test_invalid_json_returns_error(self):
        """Should return an error for non-JSON input."""
        from src.tools.analysis_tools import calculate_statistics

        result = calculate_statistics.invoke({"data_json": "not json"})
        data = json.loads(result)

        self.assertIn("error", data)

    def test_confidence_interval_present(self):
        """Statistical output should include 95% confidence interval fields."""
        from src.tools.analysis_tools import calculate_statistics

        result = calculate_statistics.invoke({"data_json": "[2, 4, 6, 8, 10, 12]"})
        data = json.loads(result)

        self.assertIn("std_dev", data)
        self.assertIn("count", data)


class TestAnalyzeEvidence(unittest.TestCase):
    """Tests for the analyze_evidence tool."""

    def test_clinical_evidence_classified_as_strong(self):
        """RCT/randomised trial mentions should yield strong evidence rating."""
        from src.tools.analysis_tools import analyze_evidence

        sources = json.dumps(
            [{"title": "RCT Study", "full_text": "A randomised controlled trial showed significant improvement."}]
        )
        result = analyze_evidence.invoke({"research_json": sources})
        data = json.loads(result)

        self.assertIn("strong", data["evidence_strength"])

    def test_returns_required_keys(self):
        """Output should always include the required schema keys."""
        from src.tools.analysis_tools import analyze_evidence

        result = analyze_evidence.invoke({"research_json": "[{}]"})
        data = json.loads(result)

        for key in ("source_count", "evidence_strength", "themes", "key_points", "methodology_types"):
            self.assertIn(key, data)

    def test_empty_sources_returns_valid_structure(self):
        """An empty source list should still return a valid structured response."""
        from src.tools.analysis_tools import analyze_evidence

        result = analyze_evidence.invoke({"research_json": "[]"})
        data = json.loads(result)

        self.assertIn("source_count", data)
        self.assertEqual(data["source_count"], 0)


# ---------------------------------------------------------------------------
# Writing Tools
# ---------------------------------------------------------------------------


class TestFormatCitation(unittest.TestCase):
    """Tests for the format_citation tool."""

    def test_apa_single_author(self):
        """APA format should include author, year, title."""
        from src.tools.writing_tools import format_citation

        source = json.dumps({
            "authors": ["Smith, J."],
            "year": 2023,
            "title": "Telemedicine in Diabetes Care",
            "journal": "Journal of Digital Health",
        })
        result = format_citation.invoke({"source_json": source, "style": "APA"})

        self.assertIn("Smith, J.", result)
        self.assertIn("2023", result)
        self.assertIn("Telemedicine in Diabetes Care", result)

    def test_vancouver_style(self):
        """Vancouver format should be numeric and include all required fields."""
        from src.tools.writing_tools import format_citation

        source = json.dumps({
            "authors": ["Jones, A.", "Brown, B."],
            "year": 2022,
            "title": "Remote Monitoring in T2D",
            "journal": "Diabetes Care",
            "volume": "45",
            "pages": "123-130",
        })
        result = format_citation.invoke({"source_json": source, "style": "Vancouver"})

        self.assertIn("Jones", result)
        self.assertIn("2022", result)
        self.assertIn("Remote Monitoring", result)

    def test_mla_style(self):
        """MLA format should include author and title."""
        from src.tools.writing_tools import format_citation

        source = json.dumps({
            "authors": ["Lee, C."],
            "year": 2021,
            "title": "Digital Health Outcomes",
            "journal": "Health Informatics Journal",
        })
        result = format_citation.invoke({"source_json": source, "style": "MLA"})

        self.assertIn("Lee", result)
        self.assertIn("Digital Health Outcomes", result)

    def test_unsupported_style_returns_error(self):
        """Should return an error message for unsupported citation styles."""
        from src.tools.writing_tools import format_citation

        result = format_citation.invoke({
            "source_json": json.dumps({"authors": ["A"], "year": 2020, "title": "T"}),
            "style": "HARVARD",
        })
        self.assertIn("Unsupported", result)


class TestExtractMarkdownSection(unittest.TestCase):
    """Tests for the extract_markdown_section tool."""

    def test_extracts_named_section(self):
        """Should extract the content of a named markdown section."""
        from src.tools.writing_tools import extract_markdown_section

        md = "# Report\n\n## Introduction\n\nSome intro text.\n\n## Methods\n\nMethods here.\n"
        result = extract_markdown_section.invoke({"markdown_text": md, "section_title": "Introduction"})

        self.assertIn("intro text", result)
        self.assertNotIn("Methods here", result)

    def test_missing_section_returns_empty_string(self):
        """Should return an empty string when the requested section does not exist."""
        from src.tools.writing_tools import extract_markdown_section

        md = "# Report\n\n## Introduction\n\nSome text.\n"
        result = extract_markdown_section.invoke({"markdown_text": md, "section_title": "NonExistentSection"})

        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# Plotly Visualisation Tool
# ---------------------------------------------------------------------------


class TestCreatePlotlyChart(unittest.TestCase):
    """Tests for the create_plotly_chart tool."""

    def _invoke(self, spec: dict) -> dict:
        from src.tools.writing_tools import create_plotly_chart
        raw = create_plotly_chart.invoke({"chart_spec_json": json.dumps(spec)})
        return json.loads(raw)

    def test_bar_chart_returns_plotly_json(self):
        """A basic bar chart spec should return a valid Plotly JSON figure."""
        result = self._invoke({
            "chart_type": "bar",
            "title": "Test Bar",
            "x_label": "Group",
            "y_label": "Count",
            "data": [{"x": ["A", "B", "C"], "y": [10, 20, 30], "name": "Series 1"}],
        })
        self.assertIn("plotly_json", result)
        fig = json.loads(result["plotly_json"])
        self.assertIn("data", fig)
        self.assertEqual(fig["data"][0]["type"], "bar")

    def test_line_chart_returns_correct_trace_type(self):
        """Line chart spec should produce a Scatter trace with lines+markers mode."""
        result = self._invoke({
            "chart_type": "line",
            "title": "Trend",
            "data": [{"x": [1, 2, 3], "y": [4, 5, 6], "name": "Trend"}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"][0]["type"], "scatter")
        self.assertIn("lines", fig["data"][0]["mode"])

    def test_pie_chart_uses_labels_and_values(self):
        """Pie chart should use 'labels' and 'values' keys from the series."""
        result = self._invoke({
            "chart_type": "pie",
            "title": "Distribution",
            "data": [{"labels": ["X", "Y", "Z"], "values": [30, 50, 20]}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"][0]["type"], "pie")
        self.assertIn("X", fig["data"][0]["labels"])

    def test_multi_series_bar_chart(self):
        """Multiple series should produce multiple traces in the figure."""
        result = self._invoke({
            "chart_type": "bar",
            "title": "Grouped",
            "data": [
                {"x": ["A", "B"], "y": [1, 2], "name": "S1"},
                {"x": ["A", "B"], "y": [3, 4], "name": "S2"},
            ],
            "layout": {"barmode": "group"},
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(len(fig["data"]), 2)
        self.assertEqual(fig["layout"]["barmode"], "group")

    def test_error_bars_passed_through(self):
        """Error bar specification should appear in the trace data."""
        result = self._invoke({
            "chart_type": "bar",
            "title": "With Errors",
            "data": [{"x": ["A", "B"], "y": [5, 8], "error_y": {"array": [0.5, 0.8]}}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertIsNotNone(fig["data"][0].get("error_y"))

    def test_heatmap_chart_type(self):
        """Heatmap chart should produce a heatmap trace with z data."""
        result = self._invoke({
            "chart_type": "heatmap",
            "title": "Correlation",
            "data": [{"x": ["A", "B"], "y": ["X", "Y"], "z": [[1, 0.5], [0.5, 1]]}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"][0]["type"], "heatmap")

    def test_box_chart_type(self):
        """Box chart should produce a box trace."""
        result = self._invoke({
            "chart_type": "box",
            "title": "Distribution",
            "data": [{"y": [1, 2, 3, 4, 5, 6], "name": "Group A"}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"][0]["type"], "box")

    def test_area_chart_fills_to_zero(self):
        """Area chart should produce a scatter trace with fill='tozeroy'."""
        result = self._invoke({
            "chart_type": "area",
            "title": "Area Chart",
            "data": [{"x": [1, 2, 3], "y": [10, 20, 15], "name": "Area"}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"][0]["fill"], "tozeroy")

    def test_waterfall_chart_type(self):
        """Waterfall chart should produce a waterfall trace."""
        result = self._invoke({
            "chart_type": "waterfall",
            "title": "Cash Flow",
            "data": [{"x": ["Q1", "Q2", "Q3"], "y": [100, -40, 60],
                       "measure": ["absolute", "relative", "relative"]}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"][0]["type"], "waterfall")

    def test_title_appears_in_layout(self):
        """The chart title should be reflected in the figure layout."""
        result = self._invoke({
            "chart_type": "bar",
            "title": "My Special Chart",
            "data": [{"x": ["A"], "y": [1]}],
        })
        fig = json.loads(result["plotly_json"])
        self.assertIn("My Special Chart", fig["layout"]["title"]["text"])

    def test_unsupported_chart_type_returns_error_string(self):
        """An unsupported chart_type should return a plain error string (not JSON)."""
        from src.tools.writing_tools import create_plotly_chart
        result = create_plotly_chart.invoke({
            "chart_spec_json": json.dumps({"chart_type": "radar", "data": []})
        })
        self.assertIn("Unsupported", result)
        self.assertIn("radar", result)

    def test_invalid_json_returns_error_string(self):
        """Malformed chart_spec_json should return a plain error string."""
        from src.tools.writing_tools import create_plotly_chart
        result = create_plotly_chart.invoke({"chart_spec_json": "not valid json"})
        self.assertIn("error", result.lower())

    def test_empty_data_produces_empty_traces(self):
        """An empty data list should still return a valid Plotly JSON with no traces."""
        result = self._invoke({"chart_type": "bar", "title": "Empty", "data": []})
        fig = json.loads(result["plotly_json"])
        self.assertEqual(fig["data"], [])


# ---------------------------------------------------------------------------
# Quality Tools
# ---------------------------------------------------------------------------


class TestCheckCompleteness(unittest.TestCase):
    """Tests for the check_completeness tool."""

    FULL_REPORT = """
# Clinical Evidence Synthesis

## Executive Summary
This report presents findings...

## Introduction
Background information...

## Methodology
We conducted a systematic review...

## Key Findings
- Finding 1
- Finding 2

## Evidence Analysis
The evidence shows...

## Discussion
Implications of the findings...

## Conclusions
In conclusion...

## Limitations
This study has limitations...

## References
1. Smith, J. (2023). Example.
"""

    def test_complete_report_scores_high(self):
        """A report with all sections should have a high completeness score."""
        from src.tools.quality_tools import check_completeness

        result = check_completeness.invoke({
            "report_text": self.FULL_REPORT,
            "content_type": "clinical",
        })
        data = json.loads(result)

        self.assertGreaterEqual(data["completeness_score"], 0.75)

    def test_empty_report_scores_zero(self):
        """An empty report should score 0."""
        from src.tools.quality_tools import check_completeness

        result = check_completeness.invoke({"report_text": "", "content_type": "general"})
        data = json.loads(result)

        self.assertEqual(data["completeness_score"], 0.0)

    def test_missing_sections_identified(self):
        """Missing sections should appear in the missing_sections list."""
        from src.tools.quality_tools import check_completeness

        minimal = "# Report\nSome content without most sections."
        result = check_completeness.invoke({"report_text": minimal, "content_type": "general"})
        data = json.loads(result)

        self.assertTrue(len(data["missing_sections"]) > 0)

    def test_returns_required_keys(self):
        """Output should always contain the required schema keys."""
        from src.tools.quality_tools import check_completeness

        result = check_completeness.invoke({"report_text": "Some text.", "content_type": "general"})
        data = json.loads(result)

        for key in ("required_sections", "present_sections", "missing_sections",
                    "completeness_score", "word_count", "issues"):
            self.assertIn(key, data)

    def test_clinical_requires_more_sections_than_general(self):
        """Clinical content type should have a larger required sections list."""
        from src.tools.quality_tools import check_completeness, CLINICAL_REQUIRED, REQUIRED_SECTIONS

        self.assertGreater(len(CLINICAL_REQUIRED), len(REQUIRED_SECTIONS))


class TestCheckRelevancy(unittest.TestCase):
    """Tests for the check_relevancy tool."""

    def test_relevant_report_scores_high(self):
        """A report containing the same keywords as the request should score high."""
        from src.tools.quality_tools import check_relevancy

        request = "telemedicine interventions diabetes management effectiveness"
        report = (
            "This report examines telemedicine interventions used in diabetes management "
            "and their effectiveness in improving patient outcomes. "
            "Telemedicine platforms have demonstrated measurable improvements in diabetes control."
        )
        result = check_relevancy.invoke({"report_text": report, "original_request": request})
        data = json.loads(result)

        self.assertGreaterEqual(data["relevancy_score"], 0.5)

    def test_irrelevant_report_scores_low(self):
        """A completely off-topic report should score lower."""
        from src.tools.quality_tools import check_relevancy

        request = "telemedicine interventions diabetes management"
        report = "The stock market experienced significant volatility due to rising inflation concerns."
        result = check_relevancy.invoke({"report_text": report, "original_request": request})
        data = json.loads(result)

        self.assertLessEqual(data["relevancy_score"], 0.4)

    def test_returns_required_keys(self):
        """Output must contain relevancy_score, matched_keywords, missing_keywords, feedback."""
        from src.tools.quality_tools import check_relevancy

        result = check_relevancy.invoke({
            "report_text": "Some report text here.",
            "original_request": "diabetes telemedicine study",
        })
        data = json.loads(result)

        for key in ("relevancy_score", "matched_keywords", "missing_keywords", "feedback"):
            self.assertIn(key, data)


class TestCheckReadability(unittest.TestCase):
    """Tests for the check_readability tool (including pure-Python fallback)."""

    SAMPLE_TEXT = (
        "The clinical trial demonstrated statistically significant improvements in glycaemic control. "
        "Patients assigned to the telemedicine intervention group achieved a mean HbA1c reduction "
        "of 0.8 percentage points compared to the control group. This difference was clinically "
        "meaningful and consistent with prior systematic reviews of remote diabetes management."
    )

    def test_returns_readability_metrics(self):
        """Should return all expected readability metric keys."""
        from src.tools.quality_tools import check_readability

        result = check_readability.invoke({"text": self.SAMPLE_TEXT})
        data = json.loads(result)

        for key in ("flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog_index",
                    "word_count", "sentence_count", "interpretation"):
            self.assertIn(key, data)

    def test_short_text_returns_error(self):
        """Text under 100 chars should return an error."""
        from src.tools.quality_tools import check_readability

        result = check_readability.invoke({"text": "Too short."})
        data = json.loads(result)

        self.assertIn("error", data)

    def test_pure_python_fallback_when_textstat_unavailable(self):
        """Should use pure-Python metrics when textstat raises any exception."""
        from src.tools.quality_tools import check_readability

        with patch.dict(sys.modules, {"textstat": None}):
            result = check_readability.invoke({"text": self.SAMPLE_TEXT})

        data = json.loads(result)

        self.assertIn("flesch_reading_ease", data)
        self.assertIn("interpretation", data)
        self.assertIn("note", data)
        self.assertIn("Pure-Python", data["note"])

    def test_interpretation_field_is_string(self):
        """interpretation field should be a non-empty string."""
        from src.tools.quality_tools import check_readability

        result = check_readability.invoke({"text": self.SAMPLE_TEXT})
        data = json.loads(result)

        self.assertIsInstance(data["interpretation"], str)
        self.assertGreater(len(data["interpretation"]), 0)


class TestCheckGrammar(unittest.TestCase):
    """Tests for the check_grammar tool (heuristic fallback)."""

    def test_heuristic_fallback_when_language_tool_unavailable(self):
        """Should use simple heuristic checking when LanguageTool is not installed."""
        from src.tools.quality_tools import check_grammar

        with patch.dict(sys.modules, {"language_tool_python": None}):
            result = check_grammar.invoke({
                "text": "This is a well-written sentence. It starts with a capital letter."
            })

        data = json.loads(result)

        self.assertIn("error_count", data)
        self.assertIn("quality", data)
        self.assertIn("issues", data)

    def test_returns_required_keys(self):
        """Output must always include error_count, quality, and issues."""
        from src.tools.quality_tools import check_grammar

        result = check_grammar.invoke({
            "text": "This sentence is fine. Another sentence here."
        })
        data = json.loads(result)

        for key in ("error_count", "quality", "issues"):
            self.assertIn(key, data)

    def test_max_errors_limits_output(self):
        """max_errors parameter should cap the number of reported issues."""
        from src.tools.quality_tools import check_grammar

        # Many short sentences that might trigger heuristic issues
        text = " ".join(["word" * 5] * 30)

        result = check_grammar.invoke({"text": text, "max_errors": 3})
        data = json.loads(result)

        self.assertLessEqual(len(data["issues"]), 3)


class TestCheckMedicalClaims(unittest.TestCase):
    """Tests for the check_medical_claims tool."""

    SOURCES = json.dumps([
        {"title": "Smith 2021", "snippet": "HbA1c reduced by 0.8% in the intervention group (p=0.03)."},
        {"title": "Jones 2022", "snippet": "Adherence improved by 23% with remote monitoring."},
    ])

    def test_verifies_claims_present_in_sources(self):
        """Numerical claims matching source data should appear as verified."""
        from src.tools.quality_tools import check_medical_claims

        report = (
            "The intervention reduced HbA1c by 0.8%. "
            "Adherence improved by 23% with remote monitoring. "
            "The p-value was 0.03, indicating statistical significance."
        )
        result = check_medical_claims.invoke({"report_text": report, "sources_json": self.SOURCES})
        data = json.loads(result)

        self.assertIn("verified_claims", data)
        self.assertIn("unverified_claims", data)
        self.assertIn("accuracy_score", data)
        self.assertGreaterEqual(data["accuracy_score"], 0.0)
        self.assertLessEqual(data["accuracy_score"], 1.0)

    def test_empty_sources_triggers_feedback(self):
        """No source text should produce a feedback warning."""
        from src.tools.quality_tools import check_medical_claims

        report = "HbA1c reduced by 0.5%. Mortality decreased by 10%."
        result = check_medical_claims.invoke({"report_text": report, "sources_json": "[]"})
        data = json.loads(result)

        self.assertTrue(any("source" in fb.lower() for fb in data["feedback"]))

    def test_returns_required_keys(self):
        """Output must contain all required schema keys."""
        from src.tools.quality_tools import check_medical_claims

        result = check_medical_claims.invoke({
            "report_text": "Mortality improved by 5% (p=0.04).",
            "sources_json": self.SOURCES,
        })
        data = json.loads(result)

        for key in ("verified_claims", "unverified_claims", "verified_count",
                    "unverified_count", "accuracy_score", "feedback"):
            self.assertIn(key, data)

    def test_invalid_sources_json_handled_gracefully(self):
        """Malformed sources_json should not crash the tool."""
        from src.tools.quality_tools import check_medical_claims

        result = check_medical_claims.invoke({
            "report_text": "Mortality reduced by 5% (p=0.04).",
            "sources_json": "not valid json",
        })
        data = json.loads(result)

        self.assertIn("accuracy_score", data)


if __name__ == "__main__":
    unittest.main()
