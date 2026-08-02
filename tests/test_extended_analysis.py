from __future__ import annotations

from pathlib import Path

import numpy as np

from usig.experiment.extended_analysis import (
    ANALYSIS_VERSION,
    SPLIT_SEEDS,
    _paired_bootstrap,
    _trivia_labels,
)
from usig.experiment.generation import load_generation_config

ROOT = Path(__file__).parents[1]


def test_generation_limits_and_prompt_version_are_configuration_driven() -> None:
    config = load_generation_config(ROOT)
    assert config["prompt_version"] == "v1"
    assert config["prompt_versions"]["gsm8k"] == "gsm8k_five_shot_concise_v4"
    assert config["max_new_tokens"]["gsm8k"] == 512
    assert config["calibration"]["sample_count"] == 100
    assert config["calibration"]["maximum_truncation_rate"] == 0.05


def test_five_split_seeds_are_fixed() -> None:
    assert SPLIT_SEEDS == (2026, 2027, 2028, 2029, 2030)
    assert ANALYSIS_VERSION == "v2"


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.asarray([0, 1] * 20)
    baseline = np.linspace(0.0, 1.0, 40)
    candidate = baseline.copy()
    first = _paired_bootstrap(labels, baseline, candidate, seed=2026, draws=100)
    second = _paired_bootstrap(labels, baseline, candidate, seed=2026, draws=100)
    assert first == second
    assert first["difference"] == 0.0
    assert first["paired_bootstrap_95_ci"] == [0.0, 0.0]


def test_trivia_label_variants_are_distinct() -> None:
    predictions = [
        {
            "binary_error": 1,
            "evaluation_diagnostics": {
                "concise_suffix": {"match": True},
                "containment": {"maximum_token_f1": 1.0},
            },
        },
        {
            "binary_error": 1,
            "evaluation_diagnostics": {
                "concise_suffix": {"match": True},
                "containment": {"maximum_token_f1": 0.5},
            },
        },
    ]
    assert _trivia_labels(predictions, "strict").tolist() == [1, 1]
    assert _trivia_labels(predictions, "alias").tolist() == [0, 0]
    assert _trivia_labels(predictions, "verified").tolist() == [0, 1]
