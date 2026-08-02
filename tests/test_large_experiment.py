from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from usig.data.large_experiment_manifests import (
    _balanced_cells,
    prepare_manifests,
    verify_manifests,
)
from usig.experiment.compact_analysis import (
    balanced_splits,
    comparison_predictions,
    fold_constant_mask,
    fold_residuals,
    gsm8k_decision,
    reject_duplicate_features,
    reliability_label,
)
from usig.experiment.large_collection import (
    COMPACT_IFI_NAMES,
    COMPACT_PROBABILITY_NAMES,
    compact_features,
)

ROOT = Path(__file__).parents[1]


def _record(
    identifier: str,
    *,
    operation: str = "addition",
    seed: int = 1,
    answerable: bool = True,
    question: str = "question",
    answer: str = "1",
) -> dict:
    return {
        "example_id": identifier,
        "group_id": identifier,
        "dataset": "fixture",
        "split": "test",
        "question": question,
        "context": identifier,
        "answerable": answerable,
        "reference_answers": [answer],
        "ambiguous": False,
        "ambiguity_count": 1,
        "metadata": {"operation": operation, "seed": seed},
    }


def test_deterministic_larger_manifest_sampling() -> None:
    records = [
        _record(f"{operation}-{seed}-{index}", operation=operation, seed=seed)
        for operation in ("addition", "subtraction")
        for seed in range(5)
        for index in range(10)
    ]
    first = _balanced_cells(
        records,
        operations=("addition", "subtraction"),
        per_operation=25,
    )
    second = _balanced_cells(
        records,
        operations=("addition", "subtraction"),
        per_operation=25,
    )
    assert [item[0]["example_id"] for item in first] == [
        item[0]["example_id"] for item in second
    ]


def test_operation_and_seed_balance() -> None:
    rows = [
        json.loads(line)
        for line in (
            ROOT
            / "data/manifests/qwen_1_5b/ifi_arith_source.jsonl"
        ).read_text().splitlines()
    ]
    operations = Counter(
        item["sampling_stratum"].split("|")[0] for item in rows
    )
    seeds = Counter(item["sampling_stratum"].split("|")[1] for item in rows)
    assert set(operations.values()) == {250}
    assert set(seeds.values()) == {200}


def test_moderate_operation_and_seed_balance() -> None:
    rows = [
        json.loads(line)
        for line in (
            ROOT
            / "data/manifests/qwen_1_5b/ifi_arith_moderate_multiplicative.jsonl"
        ).read_text().splitlines()
    ]
    operations = Counter(
        item["sampling_stratum"].split("|")[0] for item in rows
    )
    seeds = Counter(item["sampling_stratum"].split("|")[1] for item in rows)
    assert set(operations.values()) == {500}
    assert set(seeds.values()) == {200}


def test_squad_answerability_balance_and_unique_contexts() -> None:
    rows = [
        json.loads(line)
        for line in (
            ROOT / "data/manifests/qwen_1_5b/squad.jsonl"
        ).read_text().splitlines()
    ]
    labels = Counter(
        item["sampling_stratum"].split("|")[0] for item in rows
    )
    contexts = [
        item["sampling_stratum"].split("context:", 1)[1] for item in rows
    ]
    assert labels == {"answerable:true": 750, "answerable:false": 750}
    assert len(contexts) == len(set(contexts)) == 1500


def test_triviaqa_overlap_exclusion() -> None:
    index = json.loads(
        (
            ROOT / "data/manifests/qwen_1_5b/collection_index.json"
        ).read_text()
    )
    assert index["triviaqa_train_validation_overlap_exclusions"] == 57


def test_ambignq_interpretation_balance_is_preserved() -> None:
    rows = [
        json.loads(line)
        for line in (
            ROOT / "data/manifests/qwen_1_5b/ambignq.jsonl"
        ).read_text().splitlines()
    ]
    labels = {item["sampling_stratum"] for item in rows}
    assert any("ambiguous:true" in label for label in labels)
    assert any("ambiguous:false" in label for label in labels)
    assert len(labels) >= 3


def test_gsm8k_calibration_strata() -> None:
    rows = [
        json.loads(line)
        for line in (
            ROOT / "data/manifests/qwen_1_5b/gsm8k_calibration.jsonl"
        ).read_text().splitlines()
    ]
    assert len(rows) == 300
    assert len({item["sampling_stratum"] for item in rows}) == 16


def test_compact_ifi_feature_membership() -> None:
    token = {
        "mean_token_instability": 1.0,
        "maximum_token_instability": 2.0,
        "token_instability_slope": 3.0,
        "token_instability_roughness": 4.0,
    }
    layers = {
        "cosine_early_mean": 5.0,
        "cosine_middle_mean": 6.0,
        "cosine_late_mean": 7.0,
        "cosine_profile_normalized_maximum_position": 0.5,
        "cosine_profile_roughness": 8.0,
    }
    signature = {
        "feature_status": "ok",
        "scalar_ifi": 0.1,
        "cosine_token_dynamics": token,
        "cosine_structured": layers,
    }
    probability = {
        "summary": {name: float(index) for index, name in enumerate(COMPACT_PROBABILITY_NAMES)}
    }
    result = compact_features(signature, probability)
    assert tuple(result["compact_ifi"]) == COMPACT_IFI_NAMES
    assert "token_instability_std" not in result["compact_ifi"]


def test_duplicate_feature_rejection() -> None:
    matrix = np.asarray([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="duplicate"):
        reject_duplicate_features(matrix, ("first", "second"))


def test_constant_feature_removal_uses_training_fold() -> None:
    training = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    validation = np.asarray([[2.0, 2.0]])
    mask = fold_constant_mask(training)
    assert mask.tolist() == [False, True]
    assert validation[:, mask].shape == (1, 1)


def test_residualization_is_fitted_only_on_training_data() -> None:
    train_x = np.asarray([[0.0], [1.0], [2.0]])
    validation_x = np.asarray([[100.0]])
    train_ifi = np.asarray([[0.0], [1.0], [2.0]])
    validation_ifi = np.asarray([[0.0]])
    first_train, first_validation = fold_residuals(
        train_x, validation_x, train_ifi, validation_ifi
    )
    second_train, second_validation = fold_residuals(
        train_x, validation_x, train_ifi, np.asarray([[999.0]])
    )
    assert np.allclose(first_train, second_train)
    assert not np.allclose(first_validation, second_validation)


def test_validation_residual_uses_training_mapping() -> None:
    train_residual, validation_residual = fold_residuals(
        np.asarray([[0.0], [1.0], [2.0]]),
        np.asarray([[3.0]]),
        np.asarray([[0.0], [1.0], [2.0]]),
        np.asarray([[3.0]]),
    )
    assert train_residual.shape == (3, 1)
    assert validation_residual.shape == (1, 1)


def test_identical_split_reuse() -> None:
    labels = np.asarray([0, 1] * 10)
    strata = ["a", "b"] * 10
    first = balanced_splits(labels, strata, folds=2, seed=2026)
    second = balanced_splits(labels, strata, folds=2, seed=2026)
    assert [
        hashlib.sha256(validation.tobytes()).hexdigest()
        for _, validation in first
    ] == [
        hashlib.sha256(validation.tobytes()).hexdigest()
        for _, validation in second
    ]


def test_balanced_splits_distribute_unique_strata() -> None:
    labels = np.asarray([0, 1] * 10)
    strata = [f"unique-{index}" for index in range(len(labels))]
    splits = balanced_splits(labels, strata, folds=5, seed=2026)

    assert all(len(training) == 16 for training, _ in splits)
    assert all(len(validation) == 4 for _, validation in splits)
    assert all(
        sorted(labels[validation].tolist()) == [0, 0, 1, 1]
        for _, validation in splits
    )


def test_balanced_splits_change_unique_strata_across_seeds() -> None:
    labels = np.asarray([0, 1] * 20)
    strata = [f"unique-{index}" for index in range(len(labels))]
    first = balanced_splits(labels, strata, folds=5, seed=2026)
    second = balanced_splits(labels, strata, folds=5, seed=2027)
    assert [validation.tolist() for _, validation in first] != [
        validation.tolist() for _, validation in second
    ]


def test_operation_control_can_be_included() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    arrays = {
        "P": np.arange(6, dtype=float).reshape(-1, 1),
        "L": np.ones((6, 1)),
        "I": np.arange(6, dtype=float).reshape(-1, 1),
        "C": np.arange(6, dtype=float).reshape(-1, 1),
        "O": np.asarray([[0], [0], [1], [1], [0], [1]], dtype=float),
    }
    splits = balanced_splits(labels, ["a"] * 6, folds=2, seed=2026)
    scores, _ = comparison_predictions(
        arrays, labels, splits, ("P", "L", "O")
    )
    assert np.isfinite(scores).all()


def test_leave_one_operation_out_isolation() -> None:
    operations = np.asarray(["add", "add", "sub", "sub", "mul", "div"])
    held_out = np.flatnonzero(operations == "sub")
    training = np.flatnonzero(operations != "sub")
    assert not set(held_out) & set(training)
    assert set(operations[held_out]) == {"sub"}


def test_source_to_shift_isolation() -> None:
    source = {"source:a", "source:b"}
    shift = {"larger:a", "moderate:b"}
    assert source.isdisjoint(shift)


@pytest.mark.parametrize(
    ("minority", "expected"),
    [
        (19, "descriptive_only"),
        (20, "exploratory"),
        (50, "preliminary"),
        (100, "suitable_for_primary_analysis"),
    ],
)
def test_reliability_labels(minority: int, expected: str) -> None:
    assert reliability_label(minority) == expected


def test_gsm8k_decision_thresholds() -> None:
    assert "do_not" in gsm8k_decision(19)["decision"]
    assert "exploratory" in gsm8k_decision(20)["decision"]
    assert "continue" in gsm8k_decision(50)["decision"]
    assert "suitable" in gsm8k_decision(100)["decision"]


def test_class_deficient_metric_policy() -> None:
    assert reliability_label(0) == "descriptive_only"


def test_score_orientation_is_error_risk() -> None:
    source = (
        ROOT / "src/usig/experiment/compact_analysis.py"
    ).read_text()
    assert "higher_is_greater_error_risk" in source
    assert "binary_error" in source


def test_manifest_existing_output_protection() -> None:
    first = verify_manifests(ROOT)
    second = prepare_manifests(ROOT)
    assert first["counts"] == second["counts"]


def test_existing_pilot_artifact_protection() -> None:
    prediction = ROOT / "outputs/predictions/qwen_ifi_66b0032f646fc519.jsonl"
    signature = ROOT / "outputs/signatures/qwen_ifi_66b0032f646fc519.jsonl"
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == (
        "ee131679054b616852d8db5de67d2c36109a0d1a0783e613f7a17f15b6829769"
    )
    assert hashlib.sha256(signature.read_bytes()).hexdigest() == (
        "7f6050271d1e2d1136783163a44bba0b02c29bb88526dd2b7964cab9db435f9f"
    )


def test_no_parent_imports_in_large_experiment_sources() -> None:
    paths = [
        ROOT / "src/usig/data/large_experiment_manifests.py",
        ROOT / "src/usig/experiment/large_collection.py",
        ROOT / "src/usig/experiment/compact_analysis.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    assert "from ifi" not in source
    assert "import ifi" not in source
