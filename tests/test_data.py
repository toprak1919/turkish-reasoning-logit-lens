import pytest
from src.data import extract_answer_number, count_reasoning_steps, construct_prompt


class TestExtractAnswerNumber:
    def test_standard_format(self):
        assert extract_answer_number("some steps\n#### 624") == 624

    def test_with_comma(self):
        assert extract_answer_number("#### 1,234") == 1234

    def test_negative(self):
        assert extract_answer_number("#### -42") == -42

    def test_fallback_last_number(self):
        assert extract_answer_number("The answer is 99.") == 99

    def test_no_number(self):
        assert extract_answer_number("no numbers here") is None


class TestCountReasoningSteps:
    def test_simple(self):
        solution = "Step 1\nStep 2\n#### 42"
        assert count_reasoning_steps(solution) == 2

    def test_single_step(self):
        solution = "Just one step\n#### 10"
        assert count_reasoning_steps(solution) == 1


class TestConstructPrompt:
    def test_direct(self):
        prompt = construct_prompt("What is 2+2?", style="direct")
        assert "Q:" in prompt
        assert "The answer is" in prompt

    def test_cot_en(self):
        prompt = construct_prompt("What is 2+2?", style="cot_en")
        assert "step by step" in prompt

    def test_cot_tr(self):
        prompt = construct_prompt("2+2 nedir?", style="cot_tr")
        assert "adım adım" in prompt

    def test_invalid_style(self):
        with pytest.raises(ValueError):
            construct_prompt("test", style="invalid")
