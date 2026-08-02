"""Prepare and audit the controlled uncertainty-flow dry pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml

from usig.uncertainty_flow.groups import (
    SplitName,
    deterministic_group_split,
    validate_groups,
)
from usig.uncertainty_flow.schema import (
    InterventionAction,
    UncertaintyFlowRecord,
    UncertaintySource,
    UncertaintyVariant,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config/uncertainty_flow/pilot.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/uncertainty_flow/pilot_v1/manifests/dry_pilot.jsonl"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "outputs/uncertainty_flow/pilot_v1/audits/dry_pilot_audit.json"
)
DEFAULT_SPLITS = (
    PROJECT_ROOT
    / "outputs/uncertainty_flow/pilot_v1/manifests/dry_pilot_splits.json"
)


_SOURCE_DATASETS = {
    UncertaintySource.KNOWLEDGE: "synthetic_knowledge_placeholder",
    UncertaintySource.AMBIGUITY: "synthetic_ambiguity_placeholder",
    UncertaintySource.REASONING: "synthetic_reasoning_placeholder",
}

_SOURCE_ACTIONS = {
    UncertaintySource.KNOWLEDGE: InterventionAction.RETRIEVE,
    UncertaintySource.AMBIGUITY: InterventionAction.CLARIFY,
    UncertaintySource.REASONING: InterventionAction.REASON_MORE,
}


def build_dry_pilot_records(
    groups_per_source: int,
) -> list[UncertaintyFlowRecord]:
    """Build deterministic placeholder groups for schema and workflow auditing."""

    if groups_per_source <= 0:
        raise ValueError("groups_per_source must be positive")

    records: list[UncertaintyFlowRecord] = []

    for source in (
        UncertaintySource.KNOWLEDGE,
        UncertaintySource.AMBIGUITY,
        UncertaintySource.REASONING,
    ):
        dataset_name = _SOURCE_DATASETS[source]
        original_action = _SOURCE_ACTIONS[source]

        for index in range(groups_per_source):
            group_id = f"{source.value}_{index:04d}"
            base_id = f"{dataset_name}_{index:04d}"

            records.extend(
                [
                    UncertaintyFlowRecord(
                        record_id=f"{group_id}_original",
                        group_id=group_id,
                        base_id=base_id,
                        prompt=(
                            f"[PLACEHOLDER] Original {source.value} prompt "
                            f"for group {group_id}."
                        ),
                        source=source,
                        variant=UncertaintyVariant.ORIGINAL,
                        optimal_action=original_action,
                        dataset_name=dataset_name,
                        gold_answers=(f"placeholder_answer_{index:04d}",),
                        metadata={
                            "placeholder": True,
                            "manual_review_required": True,
                        },
                    ),
                    UncertaintyFlowRecord(
                        record_id=f"{group_id}_resolved",
                        group_id=group_id,
                        base_id=base_id,
                        prompt=(
                            f"[PLACEHOLDER] Resolved {source.value} prompt "
                            f"for group {group_id}."
                        ),
                        source=source,
                        variant=UncertaintyVariant.RESOLVED,
                        optimal_action=InterventionAction.ANSWER,
                        dataset_name=dataset_name,
                        gold_answers=(f"placeholder_answer_{index:04d}",),
                        is_resolved_variant=True,
                        intervention_cost=1.0,
                        metadata={
                            "placeholder": True,
                            "manual_review_required": True,
                        },
                    ),
                    UncertaintyFlowRecord(
                        record_id=f"{group_id}_irrelevant_control",
                        group_id=group_id,
                        base_id=base_id,
                        prompt=(
                            f"[PLACEHOLDER] Irrelevant-control "
                            f"{source.value} prompt for group {group_id}."
                        ),
                        source=source,
                        variant=UncertaintyVariant.IRRELEVANT_CONTROL,
                        optimal_action=original_action,
                        dataset_name=dataset_name,
                        gold_answers=(f"placeholder_answer_{index:04d}",),
                        intervention_cost=1.0,
                        metadata={
                            "placeholder": True,
                            "manual_review_required": True,
                        },
                    ),
                ]
            )

    return records


def write_jsonl(
    path: Path,
    records: Iterable[UncertaintyFlowRecord],
    assignments: dict[str, SplitName],
) -> None:
    """Write records atomically with explicit group-level split labels."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict()
            payload["split"] = assignments[record.group_id]
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    """Write one JSON artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Compute a SHA-256 checksum for one artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_config(path: Path) -> dict:
    """Load the pilot YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("pilot configuration must be a mapping")

    return payload


def prepare(
    *,
    config_path: Path,
    output_path: Path,
    audit_path: Path,
    split_path: Path,
    overwrite: bool,
) -> None:
    """Create the dry-pilot placeholder manifest and integrity audit."""

    for path in (output_path, audit_path, split_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}"
            )

    config = load_config(config_path)
    sources = config["sources"]
    groups_per_source = int(sources["knowledge"]["dry_pilot_groups"])

    expected_counts = {
        source: int(sources[source]["dry_pilot_groups"])
        for source in ("knowledge", "ambiguity", "reasoning")
    }

    if len(set(expected_counts.values())) != 1:
        raise ValueError(
            "the current dry-pilot generator requires equal group counts "
            "for knowledge, ambiguity, and reasoning"
        )

    records = build_dry_pilot_records(groups_per_source)
    audit = validate_groups(
        records,
        require_dry_pilot_variants=True,
    )

    splitting = config["splitting"]
    assignments = deterministic_group_split(
        records,
        seed=int(config["project"]["seed"]),
        train_fraction=float(splitting["train_fraction"]),
        validation_fraction=float(splitting["validation_fraction"]),
        test_fraction=float(splitting["test_fraction"]),
    )

    ordered_records = sorted(
        records,
        key=lambda record: (
            record.source.value,
            record.group_id,
            record.variant.value,
            record.record_id,
        ),
    )

    write_jsonl(output_path, ordered_records, assignments)
    write_json(
        split_path,
        {
            group_id: split.value
            for group_id, split in assignments.items()
        },
    )

    split_group_counts = {
        split.value: sum(
            assigned is split
            for assigned in assignments.values()
        )
        for split in SplitName
    }

    write_json(
        audit_path,
        {
            "project": config["project"],
            "record_count": audit.record_count,
            "group_count": audit.group_count,
            "source_counts": dict(audit.source_counts),
            "variant_counts": dict(audit.variant_counts),
            "dataset_counts": dict(audit.dataset_counts),
            "split_group_counts": split_group_counts,
            "placeholder_records": audit.record_count,
            "manual_review_required": True,
            "manifest_path": str(output_path),
            "split_path": str(split_path),
        },
    )

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["manifest_sha256"] = file_sha256(output_path)
    audit_payload["splits_sha256"] = file_sha256(split_path)
    write_json(audit_path, audit_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the uncertainty-flow dry-pilot manifest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT,
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=DEFAULT_SPLITS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    prepare(
        config_path=arguments.config,
        output_path=arguments.output,
        audit_path=arguments.audit_output,
        split_path=arguments.split_output,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
