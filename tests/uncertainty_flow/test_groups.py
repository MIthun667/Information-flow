"""Tests for counterfactual-group validation and deterministic splitting."""

from __future__ import annotations

from collections import Counter

import pytest

from usig.uncertainty_flow.groups import (
    SplitName,
    apply_group_split,
    deterministic_group_split,
    validate_groups,
)
from usig.uncertainty_flow.schema import (
    InterventionAction,
    UncertaintyFlowRecord,
    UncertaintySource,
    UncertaintyVariant,
)


def make_group(
    *,
    source: UncertaintySource,
    index: int,
    dataset_name: str,
) -> list[UncertaintyFlowRecord]:
    group_id = f"{source.value}_{index:04d}"
    base_id = f"{dataset_name}_{index:04d}"

    action_by_source = {
        UncertaintySource.KNOWLEDGE: InterventionAction.RETRIEVE,
        UncertaintySource.AMBIGUITY: InterventionAction.CLARIFY,
        UncertaintySource.REASONING: InterventionAction.REASON_MORE,
    }

    original_action = action_by_source[source]

    return [
        UncertaintyFlowRecord(
            record_id=f"{group_id}_original",
            group_id=group_id,
            base_id=base_id,
            prompt=f"Original prompt for {group_id}.",
            source=source,
            variant=UncertaintyVariant.ORIGINAL,
            optimal_action=original_action,
            dataset_name=dataset_name,
            gold_answers=("answer",),
        ),
        UncertaintyFlowRecord(
            record_id=f"{group_id}_resolved",
            group_id=group_id,
            base_id=base_id,
            prompt=f"Resolved prompt for {group_id}.",
            source=source,
            variant=UncertaintyVariant.RESOLVED,
            optimal_action=InterventionAction.ANSWER,
            dataset_name=dataset_name,
            gold_answers=("answer",),
            is_resolved_variant=True,
        ),
        UncertaintyFlowRecord(
            record_id=f"{group_id}_irrelevant",
            group_id=group_id,
            base_id=base_id,
            prompt=f"Irrelevant-control prompt for {group_id}.",
            source=source,
            variant=UncertaintyVariant.IRRELEVANT_CONTROL,
            optimal_action=original_action,
            dataset_name=dataset_name,
            gold_answers=("answer",),
        ),
    ]


def make_collection(groups_per_source: int = 10) -> list[UncertaintyFlowRecord]:
    records: list[UncertaintyFlowRecord] = []

    for source, dataset in [
        (UncertaintySource.KNOWLEDGE, "triviaqa"),
        (UncertaintySource.AMBIGUITY, "ambignq"),
        (UncertaintySource.REASONING, "ifi_arith"),
    ]:
        for index in range(groups_per_source):
            records.extend(
                make_group(
                    source=source,
                    index=index,
                    dataset_name=dataset,
                )
            )

    return records


def test_valid_dry_pilot_collection() -> None:
    audit = validate_groups(
        make_collection(),
        require_dry_pilot_variants=True,
    )

    assert audit.record_count == 90
    assert audit.group_count == 30
    assert audit.source_counts == {
        "ambiguity": 10,
        "knowledge": 10,
        "reasoning": 10,
    }
    assert audit.variant_counts == {
        "irrelevant_control": 30,
        "original": 30,
        "resolved": 30,
    }


def test_duplicate_record_ids_are_rejected() -> None:
    records = make_group(
        source=UncertaintySource.KNOWLEDGE,
        index=0,
        dataset_name="triviaqa",
    )
    records.append(records[0])

    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_groups(records)


def test_multiple_sources_in_one_group_are_rejected() -> None:
    records = make_group(
        source=UncertaintySource.KNOWLEDGE,
        index=0,
        dataset_name="triviaqa",
    )
    reasoning_record = UncertaintyFlowRecord(
        record_id="knowledge_0000_reasoning_duplicate",
        group_id="knowledge_0000",
        base_id="triviaqa_0000",
        prompt="Reasoning record with a conflicting source.",
        source=UncertaintySource.REASONING,
        variant=UncertaintyVariant.ADVERSARIAL_CONTROL,
        optimal_action=InterventionAction.REASON_MORE,
        dataset_name="triviaqa",
    )
    records.append(reasoning_record)

    with pytest.raises(ValueError, match="multiple source labels"):
        validate_groups(records)


def test_multiple_base_ids_in_one_group_are_rejected() -> None:
    records = make_group(
        source=UncertaintySource.KNOWLEDGE,
        index=0,
        dataset_name="triviaqa",
    )

    replacement = UncertaintyFlowRecord(
        record_id=records[2].record_id,
        group_id=records[2].group_id,
        base_id="different_base",
        prompt=records[2].prompt,
        source=records[2].source,
        variant=records[2].variant,
        optimal_action=records[2].optimal_action,
        dataset_name=records[2].dataset_name,
        gold_answers=records[2].gold_answers,
    )
    records[2] = replacement

    with pytest.raises(ValueError, match="multiple base IDs"):
        validate_groups(records)


def test_missing_original_is_rejected() -> None:
    records = make_group(
        source=UncertaintySource.KNOWLEDGE,
        index=0,
        dataset_name="triviaqa",
    )[1:]

    with pytest.raises(ValueError, match="no original"):
        validate_groups(records)


def test_missing_dry_pilot_variant_is_rejected() -> None:
    records = make_group(
        source=UncertaintySource.KNOWLEDGE,
        index=0,
        dataset_name="triviaqa",
    )[:2]

    with pytest.raises(ValueError, match="missing dry-pilot variants"):
        validate_groups(
            records,
            require_dry_pilot_variants=True,
        )


def test_wrong_original_action_is_rejected() -> None:
    records = make_group(
        source=UncertaintySource.AMBIGUITY,
        index=0,
        dataset_name="ambignq",
    )

    original = records[0]
    records[0] = UncertaintyFlowRecord(
        record_id=original.record_id,
        group_id=original.group_id,
        base_id=original.base_id,
        prompt=original.prompt,
        source=original.source,
        variant=original.variant,
        optimal_action=InterventionAction.RETRIEVE,
        dataset_name=original.dataset_name,
        gold_answers=original.gold_answers,
    )

    with pytest.raises(ValueError, match="does not match source"):
        validate_groups(records)


def test_resolved_variant_must_answer() -> None:
    records = make_group(
        source=UncertaintySource.REASONING,
        index=0,
        dataset_name="ifi_arith",
    )
    resolved = records[1]

    records[1] = UncertaintyFlowRecord(
        record_id=resolved.record_id,
        group_id=resolved.group_id,
        base_id=resolved.base_id,
        prompt=resolved.prompt,
        source=resolved.source,
        variant=resolved.variant,
        optimal_action=InterventionAction.REASON_MORE,
        dataset_name=resolved.dataset_name,
        gold_answers=resolved.gold_answers,
        is_resolved_variant=True,
    )

    with pytest.raises(ValueError, match="must use 'answer'"):
        validate_groups(records)


def test_split_is_deterministic() -> None:
    records = make_collection()

    first = deterministic_group_split(
        records,
        seed=2026,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    second = deterministic_group_split(
        list(reversed(records)),
        seed=2026,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    assert first == second


def test_split_is_source_stratified() -> None:
    records = make_collection()

    assignments = deterministic_group_split(
        records,
        seed=2026,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    source_by_group = {
        record.group_id: record.source
        for record in records
    }

    counts = Counter(
        (source_by_group[group_id], split)
        for group_id, split in assignments.items()
    )

    for source in (
        UncertaintySource.KNOWLEDGE,
        UncertaintySource.AMBIGUITY,
        UncertaintySource.REASONING,
    ):
        assert counts[(source, SplitName.TRAIN)] == 6
        assert counts[(source, SplitName.VALIDATION)] == 2
        assert counts[(source, SplitName.TEST)] == 2


def test_counterfactual_groups_remain_intact() -> None:
    records = make_collection()
    assignments = deterministic_group_split(
        records,
        seed=2026,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    partitions = apply_group_split(records, assignments)

    seen_group_splits: dict[str, set[SplitName]] = {}

    for split, partition in partitions.items():
        for record in partition:
            seen_group_splits.setdefault(record.group_id, set()).add(split)

    assert all(
        len(splits) == 1
        for splits in seen_group_splits.values()
    )


def test_invalid_split_fractions_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        deterministic_group_split(
            make_collection(),
            seed=2026,
            train_fraction=0.7,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
