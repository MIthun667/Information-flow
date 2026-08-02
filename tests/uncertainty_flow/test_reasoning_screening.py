"""Tests for reasoning-intervention screening utilities."""

from __future__ import annotations

import pytest
import torch

from usig.experiment.screen_reasoning_interventions import (
    classify_group,
    compute_generation_metrics,
    is_prediction_correct,
    normalize_numeric_answer,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("495", "495"),
        ("FINAL: 1,234", "1234"),
        ("final: -72", "-72"),
        ("2695\\nVerification: 2695 + 3176 = 5871", None),
        ("First 10, then 25", None),
        ("No numeric answer", None),
    ],
)
def test_normalize_numeric_answer(
    text: str,
    expected: str | None,
) -> None:
    assert normalize_numeric_answer(text) == expected


def test_numeric_correctness() -> None:
    assert is_prediction_correct("1234", ["1,234"])
    assert not is_prediction_correct("1235", ["1,234"])
    assert not is_prediction_correct(None, ["1234"])


def test_generation_metrics() -> None:
    generated = torch.tensor([1, 0])
    scores = (
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[2.0, 0.0]]),
    )

    metrics = compute_generation_metrics(
        generated_token_ids=generated,
        generation_scores=scores,
    )

    assert metrics["generated_token_count"] == 2
    assert metrics["mean_token_probability"] == pytest.approx(
        0.880797,
        rel=1e-5,
    )
    assert metrics["minimum_token_probability"] == pytest.approx(
        0.880797,
        rel=1e-5,
    )


def variant(
    *,
    correct: bool,
    probability: float,
    log_probability: float,
) -> dict:
    return {
        "is_correct": correct,
        "mean_token_probability": probability,
        "mean_token_log_probability": log_probability,
    }


def test_strong_resolving_classification() -> None:
    status, _ = classify_group(
        {
            "original": variant(
                correct=False,
                probability=0.40,
                log_probability=-1.0,
            ),
            "resolved": variant(
                correct=True,
                probability=0.80,
                log_probability=-0.2,
            ),
            "irrelevant_control": variant(
                correct=False,
                probability=0.42,
                log_probability=-0.9,
            ),
        },
        minimum_probability_gain=0.05,
        minimum_log_probability_gain=0.10,
        high_confidence_threshold=0.90,
    )

    assert status == "strong_resolving"


def test_invalid_easy_classification() -> None:
    status, _ = classify_group(
        {
            "original": variant(
                correct=True,
                probability=0.95,
                log_probability=-0.05,
            ),
            "resolved": variant(
                correct=True,
                probability=0.96,
                log_probability=-0.04,
            ),
            "irrelevant_control": variant(
                correct=True,
                probability=0.95,
                log_probability=-0.05,
            ),
        },
        minimum_probability_gain=0.05,
        minimum_log_probability_gain=0.10,
        high_confidence_threshold=0.90,
    )

    assert status == "invalid_easy"


def test_harmful_classification() -> None:
    status, _ = classify_group(
        {
            "original": variant(
                correct=True,
                probability=0.80,
                log_probability=-0.2,
            ),
            "resolved": variant(
                correct=False,
                probability=0.40,
                log_probability=-1.0,
            ),
            "irrelevant_control": variant(
                correct=True,
                probability=0.79,
                log_probability=-0.21,
            ),
        },
        minimum_probability_gain=0.05,
        minimum_log_probability_gain=0.10,
        high_confidence_threshold=0.90,
    )

    assert status == "harmful"


def test_final_marker_takes_precedence_over_other_numbers() -> None:
    text = (
        "Internal check: 2695 + 3176 = 5871\n"
        "FINAL: 2695"
    )

    assert normalize_numeric_answer(text) == "2695"
