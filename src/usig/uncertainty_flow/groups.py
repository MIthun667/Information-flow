"""Group integrity validation and leakage-safe deterministic splitting."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from usig.uncertainty_flow.schema import (
    InterventionAction,
    UncertaintyFlowRecord,
    UncertaintySource,
    UncertaintyVariant,
)


class SplitName(StrEnum):
    """Supported grouped benchmark partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class GroupAudit:
    """Summary of one validated counterfactual group."""

    group_id: str
    source: UncertaintySource
    dataset_name: str
    record_count: int
    variants: tuple[UncertaintyVariant, ...]
    record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Integrity summary for a complete uncertainty-flow collection."""

    record_count: int
    group_count: int
    source_counts: Mapping[str, int]
    variant_counts: Mapping[str, int]
    dataset_counts: Mapping[str, int]
    groups: tuple[GroupAudit, ...]


_REQUIRED_DRY_PILOT_VARIANTS = frozenset(
    {
        UncertaintyVariant.ORIGINAL,
        UncertaintyVariant.RESOLVED,
        UncertaintyVariant.IRRELEVANT_CONTROL,
    }
)


def group_records(
    records: Iterable[UncertaintyFlowRecord],
) -> dict[str, tuple[UncertaintyFlowRecord, ...]]:
    """Collect records by group ID in deterministic order."""

    grouped: dict[str, list[UncertaintyFlowRecord]] = defaultdict(list)

    for record in records:
        grouped[record.group_id].append(record)

    return {
        group_id: tuple(
            sorted(group, key=lambda item: (item.variant.value, item.record_id))
        )
        for group_id, group in sorted(grouped.items())
    }


def validate_groups(
    records: Sequence[UncertaintyFlowRecord],
    *,
    require_dry_pilot_variants: bool = False,
) -> DatasetAudit:
    """Validate IDs, source consistency, variants, and intervention semantics."""

    if not records:
        raise ValueError("uncertainty-flow collection must not be empty")

    duplicate_ids = [
        record_id
        for record_id, count in Counter(
            record.record_id for record in records
        ).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(
            "duplicate record_id values: " + ", ".join(sorted(duplicate_ids))
        )

    audits: list[GroupAudit] = []

    for group_id, group in group_records(records).items():
        _validate_group_identity(group_id, group)
        _validate_group_variants(
            group_id,
            group,
            require_dry_pilot_variants=require_dry_pilot_variants,
        )
        _validate_group_actions(group_id, group)

        audits.append(
            GroupAudit(
                group_id=group_id,
                source=group[0].source,
                dataset_name=group[0].dataset_name,
                record_count=len(group),
                variants=tuple(record.variant for record in group),
                record_ids=tuple(record.record_id for record in group),
            )
        )

    return DatasetAudit(
        record_count=len(records),
        group_count=len(audits),
        source_counts=dict(
            sorted(
                Counter(audit.source.value for audit in audits).items()
            )
        ),
        variant_counts=dict(
            sorted(
                Counter(record.variant.value for record in records).items()
            )
        ),
        dataset_counts=dict(
            sorted(
                Counter(audit.dataset_name for audit in audits).items()
            )
        ),
        groups=tuple(audits),
    )


def deterministic_group_split(
    records: Sequence[UncertaintyFlowRecord],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, SplitName]:
    """Assign complete groups to deterministic source-stratified splits."""

    _validate_split_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    audit = validate_groups(records)
    groups_by_source: dict[UncertaintySource, list[str]] = defaultdict(list)

    for group in audit.groups:
        groups_by_source[group.source].append(group.group_id)

    assignments: dict[str, SplitName] = {}

    for source, group_ids in sorted(
        groups_by_source.items(), key=lambda item: item[0].value
    ):
        ordered = sorted(
            group_ids,
            key=lambda group_id: _stable_split_key(
                seed=seed,
                source=source,
                group_id=group_id,
            ),
        )

        train_count, validation_count, _ = _allocate_split_counts(
            len(ordered),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )

        train_end = train_count
        validation_end = train_count + validation_count

        for index, group_id in enumerate(ordered):
            if index < train_end:
                split = SplitName.TRAIN
            elif index < validation_end:
                split = SplitName.VALIDATION
            else:
                split = SplitName.TEST

            assignments[group_id] = split

    _validate_split_assignment(records, assignments)
    return dict(sorted(assignments.items()))


def apply_group_split(
    records: Sequence[UncertaintyFlowRecord],
    assignments: Mapping[str, SplitName],
) -> dict[SplitName, tuple[UncertaintyFlowRecord, ...]]:
    """Materialize records by split without separating counterfactual groups."""

    _validate_split_assignment(records, assignments)

    partitions: dict[SplitName, list[UncertaintyFlowRecord]] = {
        split: [] for split in SplitName
    }

    for record in records:
        partitions[assignments[record.group_id]].append(record)

    return {
        split: tuple(
            sorted(
                partition,
                key=lambda item: (
                    item.source.value,
                    item.group_id,
                    item.variant.value,
                    item.record_id,
                ),
            )
        )
        for split, partition in partitions.items()
    }


def _validate_group_identity(
    group_id: str,
    group: Sequence[UncertaintyFlowRecord],
) -> None:
    sources = {record.source for record in group}
    datasets = {record.dataset_name for record in group}
    base_ids = {record.base_id for record in group}

    if len(sources) != 1:
        raise ValueError(
            f"group {group_id!r} contains multiple source labels: "
            f"{sorted(source.value for source in sources)}"
        )

    if len(datasets) != 1:
        raise ValueError(
            f"group {group_id!r} contains multiple datasets: "
            f"{sorted(datasets)}"
        )

    if len(base_ids) != 1:
        raise ValueError(
            f"group {group_id!r} contains multiple base IDs: "
            f"{sorted(base_ids)}"
        )


def _validate_group_variants(
    group_id: str,
    group: Sequence[UncertaintyFlowRecord],
    *,
    require_dry_pilot_variants: bool,
) -> None:
    variant_counts = Counter(record.variant for record in group)
    duplicates = [
        variant.value
        for variant, count in variant_counts.items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"group {group_id!r} contains duplicate variants: "
            f"{', '.join(sorted(duplicates))}"
        )

    if UncertaintyVariant.ORIGINAL not in variant_counts:
        raise ValueError(f"group {group_id!r} has no original variant")

    if require_dry_pilot_variants:
        missing = sorted(
            variant.value
            for variant in _REQUIRED_DRY_PILOT_VARIANTS.difference(
                variant_counts
            )
        )
        if missing:
            raise ValueError(
                f"group {group_id!r} is missing dry-pilot variants: "
                f"{', '.join(missing)}"
            )


def _validate_group_actions(
    group_id: str,
    group: Sequence[UncertaintyFlowRecord],
) -> None:
    original = next(
        record
        for record in group
        if record.variant is UncertaintyVariant.ORIGINAL
    )

    expected_action = original.expected_default_action
    if expected_action is not None and original.optimal_action is not expected_action:
        raise ValueError(
            f"group {group_id!r} original action "
            f"{original.optimal_action.value!r} does not match source "
            f"{original.source.value!r}; expected {expected_action.value!r}"
        )

    for record in group:
        if record.variant is UncertaintyVariant.RESOLVED:
            if record.optimal_action is not InterventionAction.ANSWER:
                raise ValueError(
                    f"group {group_id!r} resolved variant must use "
                    "'answer' as its optimal action"
                )

        if record.variant is UncertaintyVariant.IRRELEVANT_CONTROL:
            if record.is_resolved_variant:
                raise ValueError(
                    f"group {group_id!r} irrelevant control cannot be "
                    "marked resolved"
                )


def _stable_split_key(
    *,
    seed: int,
    source: UncertaintySource,
    group_id: str,
) -> str:
    payload = f"{seed}:{source.value}:{group_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_split_fractions(
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    fractions = (train_fraction, validation_fraction, test_fraction)

    if any(fraction < 0 or fraction > 1 for fraction in fractions):
        raise ValueError("split fractions must each lie in [0, 1]")

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")


def _allocate_split_counts(
    total: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    raw = [total * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)

    order = sorted(
        range(3),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )

    for index in order[:remainder]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def _validate_split_assignment(
    records: Sequence[UncertaintyFlowRecord],
    assignments: Mapping[str, SplitName],
) -> None:
    group_ids = {record.group_id for record in records}
    assignment_ids = set(assignments)

    missing = sorted(group_ids.difference(assignment_ids))
    extra = sorted(assignment_ids.difference(group_ids))

    if missing:
        raise ValueError(
            "missing split assignments for groups: " + ", ".join(missing)
        )

    if extra:
        raise ValueError(
            "split assignments contain unknown groups: " + ", ".join(extra)
        )

    for group_id, split in assignments.items():
        if not isinstance(split, SplitName):
            raise ValueError(
                f"group {group_id!r} has invalid split value {split!r}"
            )
