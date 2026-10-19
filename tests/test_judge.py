"""Tests for parsing and building the thread judge prompt."""

from src.ingestion.judge import judge_prompt, parse_verdicts


def test_parse_verdicts_reads_a_json_array():
    reply = '[{"resolves": true, "reason": "on topic"}]'
    assert parse_verdicts(reply, 1) == [{"resolves": True, "reason": "on topic"}]


def test_parse_verdicts_rejects_the_wrong_number_of_verdicts():
    reply = '[{"resolves": true, "reason": "on topic"}]'
    assert parse_verdicts(reply, 2) is None


def test_parse_verdicts_rejects_non_json():
    assert parse_verdicts("not json", 1) is None


def test_judge_prompt_includes_the_active_topic():
    prompt = judge_prompt(
        "what is a good strategy?",
        ["Thread 1: pumpkin tips"],
        topic="The Farmer Was Replaced",
    )
    assert "The Farmer Was Replaced" in prompt
