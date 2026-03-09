"""Unit tests for utility modules: agent_runner and logging_utils."""

import unittest
from unittest.mock import MagicMock, call, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# agent_runner — run_agent_with_forced_tools
# ---------------------------------------------------------------------------


class TestRunAgentWithForcedTools(unittest.TestCase):
    """Tests for run_agent_with_forced_tools in src/utils/agent_runner.py."""

    def _make_response(self, content: str, tool_calls: list | None = None) -> AIMessage:
        """Build a mock AIMessage using the proper constructor parameter."""
        return AIMessage(content=content, tool_calls=tool_calls or [])

    def test_first_turn_uses_forced_tool_choice(self):
        """The first LLM call must use tool_choice='any' to force tool use."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        llm = MagicMock()
        llm_forced = MagicMock()
        llm_auto = MagicMock()
        llm.bind_tools.side_effect = [llm_forced, llm_auto]

        llm_forced.invoke.return_value = self._make_response("Done")

        run_agent_with_forced_tools(llm, [], "sys", "user", "test_agent")

        llm.bind_tools.assert_any_call([], tool_choice="any")
        # Second bind_tools call should NOT have tool_choice='any'
        second_call_kwargs = llm.bind_tools.call_args_list[1]
        self.assertNotIn("tool_choice", second_call_kwargs.kwargs or {})

    def test_result_starts_with_human_message(self):
        """Result messages list should always start with the user's HumanMessage."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        llm = MagicMock()
        llm_bound = MagicMock()
        llm.bind_tools.return_value = llm_bound
        llm_bound.invoke.return_value = self._make_response("Final answer")

        result = run_agent_with_forced_tools(llm, [], "sys", "hello world", "agent")

        self.assertIsInstance(result["messages"][0], HumanMessage)
        self.assertEqual(result["messages"][0].content, "hello world")

    def test_tool_is_invoked_and_result_appended(self):
        """When the LLM returns a tool call, the tool is invoked and a ToolMessage is added."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.invoke.return_value = '{"status": "ok"}'

        llm = MagicMock()
        llm_bound = MagicMock()
        llm.bind_tools.return_value = llm_bound

        tc_response = self._make_response("", tool_calls=[
            {"name": "my_tool", "args": {"x": 42}, "id": "call_abc"}
        ])
        final_response = self._make_response("Done")
        llm_bound.invoke.side_effect = [tc_response, final_response]

        result = run_agent_with_forced_tools(llm, [mock_tool], "sys", "user", "agent")

        mock_tool.invoke.assert_called_once_with({"x": 42})
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("ok", tool_messages[0].content)

    def test_unknown_tool_returns_error_string(self):
        """Calling a tool that is not in the tool map should produce an error ToolMessage."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        llm = MagicMock()
        llm_bound = MagicMock()
        llm.bind_tools.return_value = llm_bound

        tc_response = self._make_response("", tool_calls=[
            {"name": "ghost_tool", "args": {}, "id": "call_xyz"}
        ])
        final_response = self._make_response("Done")
        llm_bound.invoke.side_effect = [tc_response, final_response]

        result = run_agent_with_forced_tools(llm, [], "sys", "user", "agent")

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("not found", tool_messages[0].content)

    def test_loop_terminates_without_tool_calls(self):
        """If the LLM returns no tool calls on the first turn, the loop exits immediately."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        llm = MagicMock()
        llm_bound = MagicMock()
        llm.bind_tools.return_value = llm_bound
        llm_bound.invoke.return_value = self._make_response("Direct answer, no tools needed.")

        result = run_agent_with_forced_tools(llm, [], "sys", "user", "agent")

        llm_bound.invoke.assert_called_once()
        # Only HumanMessage + AIMessage
        self.assertEqual(len(result["messages"]), 2)

    def test_max_iterations_respected(self):
        """Loop should not exceed max_iterations even if the LLM keeps returning tool calls."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        mock_tool = MagicMock()
        mock_tool.name = "loop_tool"
        mock_tool.invoke.return_value = "result"

        llm = MagicMock()
        llm_bound = MagicMock()
        llm.bind_tools.return_value = llm_bound

        # Always return a tool call so the loop doesn't stop naturally
        call_count = [0]

        def always_tool_call(_):
            call_count[0] += 1
            return AIMessage(content="", tool_calls=[{"name": "loop_tool", "args": {}, "id": f"c{call_count[0]}"}])

        llm_bound.invoke.side_effect = always_tool_call

        result = run_agent_with_forced_tools(llm, [mock_tool], "sys", "user", "agent", max_iterations=3)

        self.assertEqual(llm_bound.invoke.call_count, 3)

    def test_multiple_tools_in_single_turn(self):
        """Multiple tool calls in a single LLM turn should all be executed."""
        from src.utils.agent_runner import run_agent_with_forced_tools

        tool_a = MagicMock()
        tool_a.name = "tool_a"
        tool_a.invoke.return_value = "result_a"

        tool_b = MagicMock()
        tool_b.name = "tool_b"
        tool_b.invoke.return_value = "result_b"

        llm = MagicMock()
        llm_bound = MagicMock()
        llm.bind_tools.return_value = llm_bound

        tc_response = self._make_response("", tool_calls=[
            {"name": "tool_a", "args": {}, "id": "c1"},
            {"name": "tool_b", "args": {}, "id": "c2"},
        ])
        final_response = self._make_response("Done")
        llm_bound.invoke.side_effect = [tc_response, final_response]

        result = run_agent_with_forced_tools(llm, [tool_a, tool_b], "sys", "user", "agent")

        tool_a.invoke.assert_called_once()
        tool_b.invoke.assert_called_once()
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 2)


# ---------------------------------------------------------------------------
# PipelineLogger methods
# ---------------------------------------------------------------------------


class TestPipelineLoggerMethods(unittest.TestCase):
    """Tests for PipelineLogger helper methods in logging_utils."""

    def test_coordinator_dispatch_does_not_raise(self):
        """coordinator_dispatch should execute without exceptions."""
        from src.utils.logging_utils import PipelineLogger

        plogger = PipelineLogger("test")
        # Should not raise
        plogger.coordinator_dispatch("Research Agent", "Search PubMed for RCTs.")

    def test_coordinator_dispatch_long_instructions_annotated(self):
        """Instructions over 400 chars should include the total char count in output."""
        from src.utils.logging_utils import PipelineLogger, console

        plogger = PipelineLogger("test")
        long_instructions = "X" * 500

        printed_args = []
        with patch.object(console, "print", side_effect=lambda *a, **kw: printed_args.append(a)):
            plogger.coordinator_dispatch("Test Agent", long_instructions)

        combined = " ".join(
            str(getattr(a, "renderable", a)) for call in printed_args for a in call
        )
        self.assertIn("500", combined)

    def test_content_preview_truncates_and_annotates(self):
        """Content longer than max_chars should include a '... [N total chars]' annotation."""
        from src.utils.logging_utils import PipelineLogger, console

        plogger = PipelineLogger("test")
        long_content = "word " * 300  # ~1500 chars

        printed_args = []
        with patch.object(console, "print", side_effect=lambda *a, **kw: printed_args.append(a)):
            plogger.content_preview("Test Label", long_content, max_chars=100)

        combined = " ".join(
            str(getattr(a, "renderable", a)) for call in printed_args for a in call
        )
        self.assertIn("total chars", combined)

    def test_content_preview_short_content_no_annotation(self):
        """Content shorter than max_chars should NOT include the total char count annotation."""
        from src.utils.logging_utils import PipelineLogger, console

        plogger = PipelineLogger("test")
        short_content = "Short content."

        printed_args = []
        with patch.object(console, "print", side_effect=lambda *a, **kw: printed_args.append(a)):
            plogger.content_preview("Label", short_content, max_chars=200)

        combined = " ".join(
            str(getattr(a, "renderable", a)) for call in printed_args for a in call
        )
        self.assertNotIn("total chars", combined)

    def test_quality_result_does_not_raise(self):
        """quality_result should print without raising exceptions."""
        from src.utils.logging_utils import PipelineLogger

        plogger = PipelineLogger("test")
        plogger.quality_result(0.82, True, ["Good readability.", "All sections present."])

    def test_phase_start_and_end_do_not_raise(self):
        """phase_start and phase_end should execute without exceptions."""
        from src.utils.logging_utils import PipelineLogger

        plogger = PipelineLogger("test")
        plogger.phase_start("Test Phase", "Starting now")
        plogger.phase_end("Test Phase", "Completed in 1s")

    def test_phase_error_does_not_raise(self):
        """phase_error should print without raising exceptions."""
        from src.utils.logging_utils import PipelineLogger

        plogger = PipelineLogger("test")
        plogger.phase_error("Test Phase", "Something went wrong.")


# ---------------------------------------------------------------------------
# Pure-Python readability helpers (quality_tools)
# ---------------------------------------------------------------------------


class TestCountSyllables(unittest.TestCase):
    """Tests for the _count_syllables heuristic helper."""

    def test_single_syllable_words(self):
        from src.tools.quality_tools import _count_syllables
        for word in ("cat", "dog", "run", "fast"):
            self.assertEqual(_count_syllables(word), 1, f"Expected 1 syllable for '{word}'")

    def test_two_syllable_words(self):
        from src.tools.quality_tools import _count_syllables
        for word in ("hello", "table", "garden"):
            self.assertGreaterEqual(_count_syllables(word), 1)
            self.assertLessEqual(_count_syllables(word), 3)

    def test_polysyllabic_words(self):
        from src.tools.quality_tools import _count_syllables
        # "beautiful" → 3 syllables, "education" → 4 syllables
        self.assertGreaterEqual(_count_syllables("beautiful"), 3)
        self.assertGreaterEqual(_count_syllables("education"), 3)

    def test_silent_trailing_e_reduces_count(self):
        from src.tools.quality_tools import _count_syllables
        # "make" → 1 syllable (silent e)
        self.assertEqual(_count_syllables("make"), 1)

    def test_minimum_is_one(self):
        from src.tools.quality_tools import _count_syllables
        # Even a pure consonant should return at least 1
        self.assertGreaterEqual(_count_syllables("b"), 1)
        self.assertGreaterEqual(_count_syllables(""), 0)  # empty → 0

    def test_strips_punctuation(self):
        from src.tools.quality_tools import _count_syllables
        # Trailing punctuation should not affect the count
        self.assertEqual(_count_syllables("cat."), _count_syllables("cat"))
        self.assertEqual(_count_syllables("hello,"), _count_syllables("hello"))


class TestPythonReadability(unittest.TestCase):
    """Tests for the _python_readability helper function."""

    SAMPLE = (
        "The quick brown fox jumps over the lazy dog. "
        "It was a fine day for testing readability metrics. "
        "Simple sentences make scores easier to compute."
    )

    def test_returns_all_required_keys(self):
        from src.tools.quality_tools import _python_readability

        result = _python_readability(self.SAMPLE)
        for key in (
            "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog_index",
            "coleman_liau_index", "smog_index", "automated_readability_index",
            "word_count", "sentence_count",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_word_count_is_positive(self):
        from src.tools.quality_tools import _python_readability

        result = _python_readability(self.SAMPLE)
        self.assertGreater(result["word_count"], 0)

    def test_sentence_count_is_positive(self):
        from src.tools.quality_tools import _python_readability

        result = _python_readability(self.SAMPLE)
        self.assertGreater(result["sentence_count"], 0)

    def test_flesch_ease_in_plausible_range(self):
        from src.tools.quality_tools import _python_readability

        result = _python_readability(self.SAMPLE)
        # Flesch Reading Ease is typically 0–100; simple text should score fairly high
        self.assertGreater(result["flesch_reading_ease"], 40)
        self.assertLess(result["flesch_reading_ease"], 110)

    def test_single_sentence_does_not_crash(self):
        from src.tools.quality_tools import _python_readability

        result = _python_readability("This is one sentence.")
        self.assertIn("word_count", result)
        self.assertGreaterEqual(result["sentence_count"], 1)


if __name__ == "__main__":
    unittest.main()
