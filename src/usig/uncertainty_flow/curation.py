"""Loading and validation for manually curated uncertainty-flow groups."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from usig.uncertainty_flow.groups import validate_groups
from usig.uncertainty_flow.schema import (
    InterventionAction,
    UncertaintyFlowRecord,
    UncertaintySource,
    UncertaintyVariant,
)


_REQUIRED_VARIANT_KEYS = {
    "original",
    "resolved",
    "irrelevant_control",
}

_REQUIRED_AUDIT_FLAGS = {
    "single_source",
    "minimal_difference",
    "answer_unchanged",
    "resolved_intervention_valid",
    "control_non_resolving",
}


def load_curated_groups(path: Path) -> list[UncertaintyFlowRecord]:
    """Load manually curated JSONL groups and convert them to records."""

    if not path.exists():
        raise FileNotFoundError(f"curated group file does not exist: {path}")

    records: list[UncertaintyFlowRecord] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"line {line_number} must contain a JSON object"
                )

            records.extend(
                curated_group_to_records(payload, line_number=line_number)
            )

    validate_groups(
        records,
        require_dry_pilot_variants=True,
    )
    validate_curated_collection(records)

    return records


def curated_group_to_records(
    payload: Mapping[str, Any],
    *,
    line_number: int | None = None,
) -> list[UncertaintyFlowRecord]:
    """Convert one curated group object into three benchmark records."""

    location = (
        f"line {line_number}"
        if line_number is not None
        else "curated group"
    )

    required = {
        "group_id",
        "base_id",
        "dataset_name",
        "source",
        "gold_answers",
        "original",
        "resolved",
        "irrelevant_control",
        "audit",
    }

    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"{location} is missing fields: {', '.join(missing)}"
        )

    source = UncertaintySource(payload["source"])
    if source not in {
        UncertaintySource.KNOWLEDGE,
        UncertaintySource.AMBIGUITY,
        UncertaintySource.REASONING,
    }:
        raise ValueError(
            f"{location} uses unsupported dry-pilot source "
            f"{source.value!r}"
        )

    audit = payload["audit"]
    if not isinstance(audit, Mapping):
        raise ValueError(f"{location} audit must be an object")

    missing_flags = sorted(_REQUIRED_AUDIT_FLAGS.difference(audit))
    if missing_flags:
        raise ValueError(
            f"{location} audit is missing flags: "
            f"{', '.join(missing_flags)}"
        )

    failed_flags = sorted(
        flag for flag in _REQUIRED_AUDIT_FLAGS
        if audit.get(flag) is not True
    )
    if failed_flags:
        raise ValueError(
            f"{location} has failed audit flags: "
            f"{', '.join(failed_flags)}"
        )

    review_status = audit.get("review_status")
    if review_status not in {"pending", "approved", "rejected"}:
        raise ValueError(
            f"{location} has invalid review_status {review_status!r}"
        )

    variants = {
        key: payload[key]
        for key in _REQUIRED_VARIANT_KEYS
    }

    for key, variant_payload in variants.items():
        if not isinstance(variant_payload, Mapping):
            raise ValueError(
                f"{location} variant {key!r} must be an object"
            )

        prompt = variant_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{location} variant {key!r} has a blank prompt"
            )

    group_id = str(payload["group_id"])
    base_id = str(payload["base_id"])
    dataset_name = str(payload["dataset_name"])
    gold_answers = tuple(str(value) for value in payload["gold_answers"])

    metadata = {
        "curated": True,
        "review_status": review_status,
        "audit_notes": str(audit.get("notes", "")),
    }

    original = variants["original"]
    resolved = variants["resolved"]
    control = variants["irrelevant_control"]

    return [
        UncertaintyFlowRecord(
            record_id=f"{group_id}_original",
            group_id=group_id,
            base_id=base_id,
            prompt=str(original["prompt"]),
            source=source,
            variant=UncertaintyVariant.ORIGINAL,
            optimal_action=InterventionAction(
                original["optimal_action"]
            ),
            dataset_name=dataset_name,
            gold_answers=gold_answers,
            evidence=original.get("evidence"),
            clarification=original.get("clarification"),
            metadata=metadata,
        ),
        UncertaintyFlowRecord(
            record_id=f"{group_id}_resolved",
            group_id=group_id,
            base_id=base_id,
            prompt=str(resolved["prompt"]),
            source=source,
            variant=UncertaintyVariant.RESOLVED,
            optimal_action=InterventionAction(
                resolved["optimal_action"]
            ),
            dataset_name=dataset_name,
            gold_answers=gold_answers,
            is_resolved_variant=True,
            intervention_cost=1.0,
            evidence=resolved.get("evidence"),
            clarification=resolved.get("clarification"),
            metadata=metadata,
        ),
        UncertaintyFlowRecord(
            record_id=f"{group_id}_irrelevant_control",
            group_id=group_id,
            base_id=base_id,
            prompt=str(control["prompt"]),
            source=source,
            variant=UncertaintyVariant.IRRELEVANT_CONTROL,
            optimal_action=InterventionAction(
                control["optimal_action"]
            ),
            dataset_name=dataset_name,
            gold_answers=gold_answers,
            intervention_cost=1.0,
            evidence=control.get("evidence"),
            clarification=control.get("clarification"),
            metadata=metadata,
        ),
    ]


def validate_curated_collection(
    records: Iterable[UncertaintyFlowRecord],
) -> None:
    """Validate pilot balance and ensure placeholders are absent."""

    materialized = list(records)

    if any("[PLACEHOLDER]" in record.prompt for record in materialized):
        raise ValueError("curated collection contains placeholder prompts")

    group_source = {
        record.group_id: record.source
        for record in materialized
    }

    source_counts = Counter(group_source.values())
    expected = {
        UncertaintySource.KNOWLEDGE: 10,
        UncertaintySource.AMBIGUITY: 10,
        UncertaintySource.REASONING: 10,
    }

    if source_counts != expected:
        formatted = {
            source.value: source_counts.get(source, 0)
            for source in expected
        }
        raise ValueError(
            "curated dry pilot must contain exactly 10 groups per source; "
            f"found {formatted}"
        )
