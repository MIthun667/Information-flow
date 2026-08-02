from __future__ import annotations

import json
from pathlib import Path

import pytest

from usig.experiment.model_suite import (
    calibration_gate,
    prepare_calibration_manifest,
    projected_size,
    require_gate,
)


def test_projected_size_handles_balanced_and_one_class() -> None:
    assert projected_size(0.5) == 200
    assert projected_size(0.8) == 500
    assert projected_size(0.0) is None
    assert projected_size(1.0) is None


def test_calibration_manifest_is_isolated_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"example_id": f"x:{index}", "selection_order": index}) + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "models" / "qwen2_5_7b" / "manifest.jsonl"
    assert prepare_calibration_manifest(source, output)["sample_count"] == 100
    with pytest.raises(FileExistsError):
        prepare_calibration_manifest(source, output)


def test_dataset_gate_is_independent_and_enforces_projection(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "binary_error": int(index >= 50),
                    "unresolved_label": False,
                    "token_limit_reached": False,
                    "evaluation_metrics": {},
                }
            )
            + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    verification = tmp_path / "verification.json"
    artifact = {
        "valid_count": 100,
        "missing_count": 0,
        "unexpected_count": 0,
        "checksum_failure_count": 0,
        "non_finite_feature_count": 0,
    }
    verification.write_text(
        json.dumps(
            {
                "complete": True,
                "artifacts": {
                    "predictions": artifact,
                    "compact_signatures": artifact,
                    "signature_ablations": artifact,
                },
            }
        ),
        encoding="utf-8",
    )
    passed = calibration_gate(
        predictions,
        verification,
        tmp_path / "pass.json",
        dataset="squad",
        requested_records=100,
        full_available_records=1500,
    )
    assert passed["gate_passed"]
    failed = calibration_gate(
        predictions,
        verification,
        tmp_path / "fail.json",
        dataset="squad",
        requested_records=100,
        full_available_records=150,
    )
    assert not failed["gate_passed"]
    with pytest.raises(PermissionError):
        require_gate(tmp_path / "fail.json")
