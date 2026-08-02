from __future__ import annotations

import json
from pathlib import Path

import pytest

from usig.experiment.gsm8k_v4 import (
    VERSION as GSM_VERSION,
    diagnostics,
    repetition_rate,
    repeated_ngram_rate,
    repeated_sentence_rate,
    required_collection_size,
    require_gate,
    response_diagnostic,
)
from usig.experiment.repair_v3 import (
    gate_report,
    interpretation_label,
    prepare_truthfulqa_mc_manifest,
)
from usig.experiment.truthfulqa_mc import option_probabilities, ordered_options


def test_interpretation_label_construction() -> None:
    interpretations = [
        {"interpretation_id": "a", "reference_answers": ["red"]},
        {"interpretation_id": "b", "reference_answers": ["blue"]},
    ]
    wrong = interpretation_label("green", interpretations)
    partial = interpretation_label("red", interpretations)
    complete = interpretation_label("red; blue", interpretations)
    assert wrong["label"] == "incorrect"
    assert partial["label"] == "partially_correct"
    assert complete["label"] == "fully_correct"
    assert wrong["fully_wrong_target"] == 1
    assert partial["fully_wrong_target"] == 0
    assert complete["incomplete_target"] == 0


def test_truthfulqa_manifest_is_deterministic_and_isolated(tmp_path: Path) -> None:
    normalized = tmp_path / "truth.jsonl"
    record = {
        "example_id": "truthfulqa:all:1",
        "group_id": "question:1",
        "category": "science",
        "reference_answers": ["correct"],
        "incorrect_reference_answers": ["wrong one", "wrong two"],
        "metadata": {
            "best_answer": "correct",
            "adversarial_type": "Adversarial",
        },
    }
    normalized.write_text(json.dumps(record) + "\n", encoding="utf-8")
    first = tmp_path / "repair-a" / "manifest.jsonl"
    second = tmp_path / "repair-b" / "manifest.jsonl"
    prepare_truthfulqa_mc_manifest(normalized, first)
    prepare_truthfulqa_mc_manifest(normalized, second)
    assert first.read_text() == second.read_text()
    row = json.loads(first.read_text())
    assert row["option_count"] == 3
    with pytest.raises(FileExistsError):
        prepare_truthfulqa_mc_manifest(normalized, first)


def test_truthfulqa_option_probabilities_and_order() -> None:
    record = {
        "example_id": "truthfulqa:all:1",
        "reference_answers": ["good", "also good"],
        "incorrect_reference_answers": ["bad a", "bad b"],
        "metadata": {"best_answer": "good"},
    }
    first = ordered_options(record)
    second = ordered_options(record)
    assert first == second
    assert sum(item["correct"] for item in first) == 2
    assert sum(item["mc1_correct"] for item in first) == 1
    probabilities = option_probabilities([-1.0, -2.0, -3.0])
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert sum(probabilities) == pytest.approx(1.0)


def test_calibration_gate_reports_classes_and_rejects_deficiency(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "binary_error": int(index >= 90),
            "unresolved_label": False,
        }
        for index in range(100)
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = gate_report(predictions, tmp_path / "gate.json", expected_count=100)
    assert report["class_counts"] == {"correct": 90, "incorrect": 10}
    assert report["reliability_status"] == "descriptive_only"
    assert report["passed"] is False


def _gsm_prediction(
    *, truncated: bool = False, parsed: bool = True, correct: bool = True
) -> dict:
    return {
        "example_id": "gsm:1",
        "response": "work\nFinal answer: 12",
        "generated_token_ids": [1, 2, 3, 3],
        "generated_token_count": 4,
        "response_character_count": 21,
        "token_limit_reached": truncated,
        "stop_reason": "generation_stopped",
        "final_answer_stop_detected": True,
        "evaluation_metrics": {
            "parsing_status": "ok" if parsed else "error",
            "exact_match": correct,
        },
    }


def test_gsm8k_diagnostics_and_gate(tmp_path: Path) -> None:
    item = response_diagnostic(_gsm_prediction())
    assert item["answer_marker_present"] is True
    assert item["final_answer_stop_detected"] is True
    assert repetition_rate([1, 1, 2, 2]) == 0.5
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(_gsm_prediction(correct=index < 50)) + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    report = diagnostics(predictions, tmp_path / "diagnostics.json")
    assert report["version"] == GSM_VERSION
    assert report["passed"] is True
    assert report["projected_required_collection_size"] == 200


def test_gsm8k_repetition_and_required_sample_diagnostics() -> None:
    assert repeated_ngram_rate("a b c d a b c d") > 0
    assert repeated_sentence_rate("Again. Again. Again.") > 0
    assert required_collection_size(0.5) == 200
    assert required_collection_size(0.0) is None
    assert required_collection_size(1.0) is None


def test_invalid_gsm8k_gate_is_rejected(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"version": GSM_VERSION, "passed": False}), encoding="utf-8"
    )
    with pytest.raises(PermissionError):
        require_gate(gate)
