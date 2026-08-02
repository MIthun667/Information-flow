"""Tests for real IFI-ARITH reasoning-group preparation."""

from __future__ import annotations

from usig.experiment.prepare_reasoning_curated_pilot import (
    TARGET_OPERATION_COUNTS,
    reasoning_scaffold,
    select_candidates,
    transform_record,
)


def source_record(
    operation: str,
    index: int,
) -> dict:
    symbols = {
        "addition": "+",
        "subtraction": "-",
        "multiplication": "×",
        "division": "÷",
    }
    operand_a = 9000 + index
    operand_b = 10 + index

    return {
        "example_id": f"ifi_arith:test:{operation}:{index:04d}",
        "source_id": f"source_{index}",
        "dataset": "ifi_arith",
        "domain": "test_domain",
        "question": f"{operand_a} {symbols[operation]} {operand_b}",
        "reference_answers": [str(index)],
        "metadata": {
            "operand_a": operand_a,
            "operand_b": operand_b,
            "operation": operation,
            "expression": (
                f"{operand_a} {symbols[operation]} {operand_b}"
            ),
            "expected_answer": index,
            "normalized_question_hash": f"hash_{index}",
        },
    }


def test_operation_scaffolds_do_not_include_answer() -> None:
    for operation in TARGET_OPERATION_COUNTS:
        scaffold = reasoning_scaffold(
            operation=operation,
            operand_a=9746,
            operand_b=4237,
        )
        assert scaffold
        assert "13983" not in scaffold


def test_selection_uses_target_operation_counts() -> None:
    records = [
        source_record(operation, index)
        for operation in TARGET_OPERATION_COUNTS
        for index in range(10)
    ]

    selected = select_candidates(records)

    counts = {
        operation: sum(
            record["metadata"]["operation"] == operation
            for record in selected
        )
        for operation in TARGET_OPERATION_COUNTS
    }

    assert counts == TARGET_OPERATION_COUNTS
    assert len(selected) == 10


def test_transformed_group_preserves_source_identity() -> None:
    source = source_record("multiplication", 3)

    group = transform_record(source, group_index=0)

    assert group["group_id"] == "reasoning_0000"
    assert group["base_id"] == source["example_id"]
    assert group["gold_answers"] == source["reference_answers"]
    assert group["source"] == "reasoning"
    assert group["audit"]["review_status"] == "pending"
    assert group["provenance"]["operation"] == "multiplication"


def test_resolved_and_control_preserve_question() -> None:
    source = source_record("division", 2)

    group = transform_record(source, group_index=2)

    question = source["question"]

    assert question in group["original"]["prompt"]
    assert question in group["resolved"]["prompt"]
    assert question in group["irrelevant_control"]["prompt"]
    assert group["resolved"]["prompt"] != group["original"]["prompt"]
    assert (
        group["irrelevant_control"]["prompt"]
        != group["original"]["prompt"]
    )
