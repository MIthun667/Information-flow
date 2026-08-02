"""Tests for uncertainty-flow record validation."""

from __future__ import annotations

import pytest

from usig.uncertainty_flow.schema import (
    InterventionAction,
    UncertaintyFlowRecord,
    UncertaintySource,
    UncertaintyVariant,
    canonical_action_for_source,
)


def make_record(**overrides: object) -> UncertaintyFlowRecord:
    payload: dict[str, object] = {
        "record_id": "knowledge_0001_original",
        "group_id": "knowledge_0001",
        "base_id": "triviaqa_123",
        "prompt": "Who discovered the example principle?",
        "source": UncertaintySource.KNOWLEDGE,
        "variant": UncertaintyVariant.ORIGINAL,
        "optimal_action": InterventionAction.RETRIEVE,
        "dataset_name": "triviaqa",
        "gold_answers": ("Example Scientist",),
    }
    payload.update(overrides)
    return UncertaintyFlowRecord(**payload)  # type: ignore[arg-type]


def test_valid_record() -> None:
    record = make_record()

    assert record.source is UncertaintySource.KNOWLEDGE
    assert record.expected_default_action is InterventionAction.RETRIEVE
    assert not record.is_control


def test_round_trip_serialization() -> None:
    original = make_record(
        evidence="A relevant evidence passage.",
        intervention_cost=1.5,
        metadata={"split": "pilot"},
    )

    reconstructed = UncertaintyFlowRecord.from_dict(original.to_dict())

    assert reconstructed == original


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (UncertaintySource.LOW_UNCERTAINTY, InterventionAction.ANSWER),
        (UncertaintySource.KNOWLEDGE, InterventionAction.RETRIEVE),
        (UncertaintySource.AMBIGUITY, InterventionAction.CLARIFY),
        (UncertaintySource.REASONING, InterventionAction.REASON_MORE),
        (UncertaintySource.MIXED, None),
    ],
)
def test_canonical_actions(
    source: UncertaintySource,
    expected: InterventionAction | None,
) -> None:
    assert canonical_action_for_source(source) is expected


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("record_id", ""),
        ("group_id", " "),
        ("base_id", ""),
        ("prompt", "\t"),
        ("dataset_name", ""),
    ],
)
def test_blank_required_fields_are_rejected(
    field_name: str,
    field_value: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        make_record(**{field_name: field_value})


def test_negative_intervention_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_record(intervention_cost=-0.1)


def test_original_cannot_be_marked_resolved() -> None:
    with pytest.raises(ValueError, match="original variant"):
        make_record(is_resolved_variant=True)


def test_resolved_variant_requires_resolved_flag() -> None:
    with pytest.raises(ValueError, match="resolved variant"):
        make_record(
            variant=UncertaintyVariant.RESOLVED,
            is_resolved_variant=False,
        )


def test_resolved_variant_is_valid() -> None:
    record = make_record(
        record_id="knowledge_0001_resolved",
        variant=UncertaintyVariant.RESOLVED,
        is_resolved_variant=True,
        optimal_action=InterventionAction.ANSWER,
        evidence="The relevant fact is supplied here.",
    )

    assert record.is_resolved_variant


def test_irrelevant_variant_is_control() -> None:
    record = make_record(
        record_id="knowledge_0001_irrelevant",
        variant=UncertaintyVariant.IRRELEVANT_CONTROL,
        evidence="An unrelated but similarly sized passage.",
    )

    assert record.is_control


def test_blank_gold_answer_is_rejected() -> None:
    with pytest.raises(ValueError, match="blank answers"):
        make_record(gold_answers=("Valid", " "))


def test_mixed_source_requires_metadata() -> None:
    with pytest.raises(ValueError, match="mixed_sources"):
        make_record(
            source=UncertaintySource.MIXED,
            optimal_action=InterventionAction.ABSTAIN,
        )


def test_mixed_source_requires_two_distinct_sources() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_record(
            source=UncertaintySource.MIXED,
            optimal_action=InterventionAction.ABSTAIN,
            metadata={"mixed_sources": ["knowledge"]},
        )


def test_valid_mixed_source_record() -> None:
    record = make_record(
        source=UncertaintySource.MIXED,
        optimal_action=InterventionAction.CLARIFY,
        metadata={
            "mixed_sources": [
                UncertaintySource.KNOWLEDGE.value,
                UncertaintySource.AMBIGUITY.value,
            ]
        },
    )

    assert record.expected_default_action is None


def test_missing_required_mapping_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required"):
        UncertaintyFlowRecord.from_dict({"record_id": "x"})
