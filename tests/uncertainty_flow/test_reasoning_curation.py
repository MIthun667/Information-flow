"""Tests for quota-controlled IFI-ARITH reasoning candidate selection."""

from __future__ import annotations

from collections import Counter

from usig.experiment.prepare_reasoning_curated_pilot import (
    EXPECTED_TOTAL,
    SELECTION_QUOTAS,
    reasoning_scaffold,
    select_candidates,
    transform_record,
    triviality_reasons,
)


def source_record(
    *,
    domain: str,
    operation: str,
    index: int,
) -> dict:
    symbols = {
        "addition": "+",
        "subtraction": "-",
        "multiplication": "×",
        "division": "÷",
    }

    if operation == "addition":
        operand_a = 8123 + index
        operand_b = 4678 + index
        answer = operand_a + operand_b
    elif operation == "subtraction":
        operand_a = 9876 + (index * 3)
        operand_b = 4321 + index
        answer = operand_a - operand_b
    elif operation == "multiplication":
        operand_a = 347 + index
        operand_b = 286 + index
        answer = operand_a * operand_b
    else:
        operand_b = 37 + index
        answer = 421 + index
        operand_a = operand_b * answer

    expression = f"{operand_a} {symbols[operation]} {operand_b}"

    return {
        "example_id": f"ifi_arith:{domain}:{operation}:{index:04d}",
        "source_id": f"source_{domain}_{operation}_{index}",
        "domain": domain,
        "question": expression,
        "reference_answers": [str(answer)],
        "metadata": {
            "operand_a": operand_a,
            "operand_b": operand_b,
            "operation": operation,
            "expression": expression,
            "expected_answer": answer,
            "normalized_question_hash": f"hash_{index}",
        },
    }


def make_candidate_pool() -> dict[str, list[dict]]:
    pool = {
        "larger_integer": [],
        "moderate_multiplicative": [],
    }

    for domain, operation, required in SELECTION_QUOTAS:
        for index in range(required + 5):
            pool[domain].append(
                source_record(
                    domain=domain,
                    operation=operation,
                    index=index + 20,
                )
            )

    return pool


def test_trivial_multiplication_is_rejected() -> None:
    record = source_record(
        domain="moderate_multiplicative",
        operation="multiplication",
        index=1,
    )
    record["metadata"]["operand_b"] = 999

    assert "convenient_multiplier" in triviality_reasons(record)


def test_small_subtraction_difference_is_rejected() -> None:
    record = source_record(
        domain="larger_integer",
        operation="subtraction",
        index=1,
    )
    record["metadata"]["operand_a"] = 5000
    record["metadata"]["operand_b"] = 4990
    record["metadata"]["expected_answer"] = 10

    assert "small_difference" in triviality_reasons(record)


def test_selection_obeys_all_quotas() -> None:
    selected, audit = select_candidates(make_candidate_pool())

    counts = Counter(
        (
            record["domain"],
            record["metadata"]["operation"],
        )
        for record in selected
    )

    assert len(selected) == EXPECTED_TOTAL

    for domain, operation, expected_count in SELECTION_QUOTAS:
        assert counts[(domain, operation)] == expected_count

    assert "selection_quotas" in audit


def test_selected_answers_are_unique() -> None:
    selected, _ = select_candidates(make_candidate_pool())

    answers = [
        record["metadata"]["expected_answer"]
        for record in selected
    ]

    assert len(answers) == len(set(answers))


def test_scaffold_is_instance_aware_and_answer_free() -> None:
    record = source_record(
        domain="moderate_multiplicative",
        operation="multiplication",
        index=31,
    )

    scaffold = reasoning_scaffold(record)

    assert str(record["metadata"]["operand_a"]) in scaffold
    assert str(record["metadata"]["operand_b"]) in scaffold
    assert str(record["metadata"]["expected_answer"]) not in scaffold


def test_transformation_preserves_identity_and_answer() -> None:
    record = source_record(
        domain="larger_integer",
        operation="addition",
        index=33,
    )

    group = transform_record(record, group_index=0)

    assert group["group_id"] == "reasoning_0000"
    assert group["base_id"] == record["example_id"]
    assert group["gold_answers"] == record["reference_answers"]
    assert group["audit"]["review_status"] == "pending"
    assert group["provenance"]["triviality_reasons"] == []


def test_all_variants_preserve_the_expression() -> None:
    record = source_record(
        domain="larger_integer",
        operation="division",
        index=27,
    )

    group = transform_record(record, group_index=2)
    question = record["question"]

    assert question in group["original"]["prompt"]
    assert question in group["resolved"]["prompt"]
    assert question in group["irrelevant_control"]["prompt"]


def test_repeated_digits_alone_do_not_reject_multiplication() -> None:
    record = source_record(
        domain="moderate_multiplicative",
        operation="multiplication",
        index=41,
    )
    record["metadata"]["operand_a"] = 777
    record["metadata"]["operand_b"] = 286
    record["metadata"]["expected_answer"] = 777 * 286
    record["metadata"]["expression"] = "777 × 286"
    record["question"] = "777 × 286"
    record["reference_answers"] = [str(777 * 286)]

    reasons = triviality_reasons(record)

    assert "repeated_digit_operand_a" not in reasons
    assert "convenient_multiplier" not in reasons
