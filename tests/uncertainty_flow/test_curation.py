"""Tests for manually curated uncertainty-flow groups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usig.uncertainty_flow.curation import (
    curated_group_to_records,
    load_curated_groups,
)
from usig.uncertainty_flow.schema import UncertaintyVariant


def valid_group(
    source: str = "knowledge",
    index: int = 0,
) -> dict:
    action = {
        "knowledge": "retrieve",
        "ambiguity": "clarify",
        "reasoning": "reason_more",
    }[source]

    return {
        "group_id": f"{source}_{index:04d}",
        "base_id": f"dataset_{index:04d}",
        "dataset_name": f"test_{source}",
        "source": source,
        "gold_answers": ["answer"],
        "original": {
            "prompt": f"Original {source} prompt {index}.",
            "optimal_action": action,
            "evidence": None,
            "clarification": None,
        },
        "resolved": {
            "prompt": f"Resolved {source} prompt {index}.",
            "optimal_action": "answer",
            "evidence": "valid evidence" if source == "knowledge" else None,
            "clarification": (
                "valid clarification"
                if source == "ambiguity"
                else None
            ),
        },
        "irrelevant_control": {
            "prompt": f"Control {source} prompt {index}.",
            "optimal_action": action,
            "evidence": "irrelevant evidence",
            "clarification": None,
        },
        "audit": {
            "single_source": True,
            "minimal_difference": True,
            "answer_unchanged": True,
            "resolved_intervention_valid": True,
            "control_non_resolving": True,
            "review_status": "approved",
            "notes": "",
        },
    }


def test_curated_group_conversion() -> None:
    records = curated_group_to_records(valid_group())

    assert len(records) == 3
    assert {
        record.variant for record in records
    } == {
        UncertaintyVariant.ORIGINAL,
        UncertaintyVariant.RESOLVED,
        UncertaintyVariant.IRRELEVANT_CONTROL,
    }


def test_failed_audit_flag_is_rejected() -> None:
    payload = valid_group()
    payload["audit"]["minimal_difference"] = False

    with pytest.raises(ValueError, match="failed audit flags"):
        curated_group_to_records(payload)


def test_invalid_review_status_is_rejected() -> None:
    payload = valid_group()
    payload["audit"]["review_status"] = "unknown"

    with pytest.raises(ValueError, match="invalid review_status"):
        curated_group_to_records(payload)


def test_loader_requires_ten_groups_per_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "groups.jsonl"
    path.write_text(
        json.dumps(valid_group()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 10 groups"):
        load_curated_groups(path)


def test_complete_curated_collection(tmp_path: Path) -> None:
    path = tmp_path / "groups.jsonl"

    groups = [
        valid_group(source, index)
        for source in ("knowledge", "ambiguity", "reasoning")
        for index in range(10)
    ]

    path.write_text(
        "".join(json.dumps(group) + "\n" for group in groups),
        encoding="utf-8",
    )

    records = load_curated_groups(path)

    assert len(records) == 90
    assert len({record.group_id for record in records}) == 30


def test_placeholder_prompt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "groups.jsonl"

    groups = [
        valid_group(source, index)
        for source in ("knowledge", "ambiguity", "reasoning")
        for index in range(10)
    ]
    groups[0]["original"]["prompt"] = "[PLACEHOLDER] invalid"

    path.write_text(
        "".join(json.dumps(group) + "\n" for group in groups),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="placeholder"):
        load_curated_groups(path)
