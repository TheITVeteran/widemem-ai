"""Judge/answerer separation and answer-cap contract for the LoCoMo harnesses.

Per benchmark/HONEST_LOCOMO.md, a published number needs a judge model that
differs from the answerer. Both harnesses read WM_JUDGE_MODEL; the default
stays the answerer model (unchanged behavior for dev-loop runs).

The answer prompts must not cap count/list answers at 5-6 words: multi-hop
aggregation questions need the complete count or every matching item.
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))


def _fresh_import(name, monkeypatch, judge_env=None):
    if judge_env is None:
        monkeypatch.delenv("WM_JUDGE_MODEL", raising=False)
    else:
        monkeypatch.setenv("WM_JUDGE_MODEL", judge_env)
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_mini_locomo_judge_defaults_to_answerer(monkeypatch):
    mod = _fresh_import("mini_locomo", monkeypatch)
    assert mod.JUDGE_LLM == mod.EVAL_LLM


def test_mini_locomo_judge_env_override(monkeypatch):
    mod = _fresh_import("mini_locomo", monkeypatch, judge_env="claude-sonnet-4-6")
    assert mod.JUDGE_LLM == "claude-sonnet-4-6"
    assert mod.EVAL_LLM != mod.JUDGE_LLM


def test_val_judge_defaults_to_answerer(monkeypatch):
    mod = _fresh_import("val", monkeypatch)
    assert mod.JUDGE_LLM == mod.EVAL_LLM


def test_val_judge_env_override(monkeypatch):
    mod = _fresh_import("val", monkeypatch, judge_env="claude-sonnet-4-6")
    assert mod.JUDGE_LLM == "claude-sonnet-4-6"


def test_mini_locomo_answer_prompt_allows_complete_lists(monkeypatch):
    mod = _fresh_import("mini_locomo", monkeypatch)
    p = mod.ANSWER_PROMPT.lower()
    assert "complete count" in p or "every matching item" in p
    # the 5-6 word cap must be conditional ("otherwise"), never unconditional
    assert "8. the answer should be less than 5-6 words" not in p
    assert "otherwise, the answer should be less than 5-6 words" in p


def test_val_open_domain_prompt_allows_complete_lists(monkeypatch):
    mod = _fresh_import("val", monkeypatch)
    p = mod.OPEN_DOMAIN_ANSWER_PROMPT.lower()
    assert "complete count" in p or "every matching item" in p


def test_honest_core_full_context_prompt_allows_complete_lists(monkeypatch):
    mod = _fresh_import("honest_core", monkeypatch)
    p = mod.FULL_CONTEXT_PROMPT.lower()
    assert "count or a list" in p
    assert "not mentioned" in p
