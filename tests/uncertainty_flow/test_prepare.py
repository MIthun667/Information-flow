"""Tests for the uncertainty-flow dry-pilot generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from usig.experiment.prepare_uncertainty_flow import prepare


def write_config(path: Path) -> None:
    payload = {
        "project": {
            "name": "uncertainty_flow",
            "version": "test_v1",
            "seed": 2026,
        },
        "sources": {
            "knowledge": {"dry_pilot_groups": 10},
            "ambiguity": {"dry_pilot_groups": 10},
            "reasoning": {"dry_pilot_groups": 10},
        },
        "splitting": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
        },
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def test_prepare_writes_valid_dry_pilot(tmp_path: Path) -> None:
    config = tmp_path / "pilot.yaml"
    manifest = tmp_path / "dry_pilot.jsonl"
    audit = tmp_path / "audit.json"
    splits = tmp_path / "splits.json"

    write_config(config)

    prepare(
        config_path=config,
        output_path=manifest,
        audit_path=audit,
        split_path=splits,
        overwrite=False,
    )

    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    split_payload = json.loads(splits.read_text(encoding="utf-8"))

    assert len(records) == 90
    assert len(split_payload) == 30
    assert audit_payload["record_count"] == 90
    assert audit_payload["group_count"] == 30
    assert audit_payload["source_counts"] == {
        "ambiguity": 10,
        "knowledge": 10,
        "reasoning": 10,
    }
    assert audit_payload["split_group_counts"] == {
        "test": 6,
        "train": 18,
        "validation": 6,
    }
    assert audit_payload["manual_review_required"] is True
    assert all(record["metadata"]["placeholder"] for record in records)


def test_prepare_refuses_overwrite(tmp_path: Path) -> None:
    config = tmp_path / "pilot.yaml"
    manifest = tmp_path / "dry_pilot.jsonl"
    audit = tmp_path / "audit.json"
    splits = tmp_path / "splits.json"

    write_config(config)

    prepare(
        config_path=config,
        output_path=manifest,
        audit_path=audit,
        split_path=splits,
        overwrite=False,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(
            config_path=config,
            output_path=manifest,
            audit_path=audit,
            split_path=splits,
            overwrite=False,
        )


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    config = tmp_path / "pilot.yaml"
    write_config(config)

    first_manifest = tmp_path / "first.jsonl"
    first_audit = tmp_path / "first_audit.json"
    first_splits = tmp_path / "first_splits.json"

    second_manifest = tmp_path / "second.jsonl"
    second_audit = tmp_path / "second_audit.json"
    second_splits = tmp_path / "second_splits.json"

    prepare(
        config_path=config,
        output_path=first_manifest,
        audit_path=first_audit,
        split_path=first_splits,
        overwrite=False,
    )
    prepare(
        config_path=config,
        output_path=second_manifest,
        audit_path=second_audit,
        split_path=second_splits,
        overwrite=False,
    )

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first_splits.read_bytes() == second_splits.read_bytes()
