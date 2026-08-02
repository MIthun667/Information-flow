"""Tests for model-conditioned reasoning candidate discovery."""

from __future__ import annotations

from collections import Counter

import pytest

from usig.experiment.discover_reasoning_candidates import (
    classify_original_record,
    parse_sampling_quotas,
    select_discovery_pool,
    select_shortlist,
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
        operand_a = 7103 + index
        operand_b = 4827 + (index * 2)
        answer = operand_a + operand_b

    elif operation == "subtraction":
        operand_a = 9821 + (index * 3)
        operand_b = 4317 + index
        answer = operand_a - operand_b

    elif operation == "multiplication":
        operand_a = 347 + index
        operand_b = 286 + (index * 2)
        answer = operand_a * operand_b

    elif operation == "division":
        operand_b = 37 + index
        answer = 421 + (index * 2)
        operand_a = operand_b * answer

    else:
        raise ValueError(operation)

    expression = (
        f"{operand_a} {symbols[operation]} {operand_b}"
    )

    return {
        "example_id": (
            f"ifi_arith:{domain}:{operation}:{index:04d}"
        ),
        "source_id": f"source_{index}",
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


def test_parse_sampling_quotas() -> None:
    quotas = parse_sampling_quotas(
        [
            {
                "domain": "larger_integer",
                "operation": "addition",
                "count": 10,
            },
            {
                "domain": "larger_integer",
                "operation": "division",
                "count": 5,
            },
        ]
    )

    assert quotas == [
        ("larger_integer", "addition", 10),
        ("larger_integer", "division", 5),
    ]


def test_duplicate_sampling_bucket_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_sampling_quotas(
            [
                {
                    "domain": "larger_integer",
                    "operation": "addition",
                    "count": 2,
                },
                {
                    "domain": "larger_integer",
                    "operation": "addition",
                    "count": 3,
                },
            ]
        )


def test_discovery_pool_obeys_quotas() -> None:
    records = {
        "larger_integer": [
            *[
                source_record(
                    domain="larger_integer",
                    operation="addition",
                    index=index + 20,
                )
                for index in range(10)
            ],
            *[
                source_record(
                    domain="larger_integer",
                    operation="division",
                    index=index + 40,
                )
                for index in range(10)
            ],
        ]
    }

    selected, audit = select_discovery_pool(
        records,
        [
            ("larger_integer", "addition", 4),
            ("larger_integer", "division", 3),
        ],
    )

    counts = Counter(
        (
            record["domain"],
            record["metadata"]["operation"],
        )
        for record in selected
    )

    assert len(selected) == 7
    assert counts[("larger_integer", "addition")] == 4
    assert counts[("larger_integer", "division")] == 3
    assert audit["selected_count"] == 7


@pytest.mark.parametrize(
    ("correct", "gold_probability", "expected"),
    [
        (False, 0.99, "wrong"),
        (True, 0.40, "low_gold_likelihood"),
        (True, 0.95, "easy"),
        (True, None, "missing_gold_score"),
    ],
)
def test_classify_original_record(
    correct: bool,
    gold_probability: float | None,
    expected: str,
) -> None:
    record = {
        "is_correct": correct,
        "gold_mean_token_probability": gold_probability,
    }

    assert classify_original_record(
        record,
        low_gold_probability_threshold=0.80,
    ) == expected


def screening_record(
    *,
    example_id: str,
    bucket: str,
    status: str,
    gold_log_probability: float,
) -> dict:
    return {
        "example_id": example_id,
        "bucket": bucket,
        "discovery_status": status,
        "gold_mean_token_log_probability": gold_log_probability,
        "mean_token_entropy": 0.5,
    }


def test_shortlist_prioritizes_wrong_examples() -> None:
    records = [
        screening_record(
            example_id="wrong_1",
            bucket="a:addition",
            status="wrong",
            gold_log_probability=-3.0,
        ),
        screening_record(
            example_id="uncertain_1",
            bucket="a:addition",
            status="low_gold_likelihood",
            gold_log_probability=-4.0,
        ),
        screening_record(
            example_id="easy_1",
            bucket="a:addition",
            status="easy",
            gold_log_probability=-0.01,
        ),
    ]

    selected = select_shortlist(
        records,
        shortlist_size=2,
        minimum_per_bucket=0,
        maximum_per_bucket=2,
    )

    assert [
        record["example_id"]
        for record in selected
    ] == [
        "wrong_1",
        "uncertain_1",
    ]


def test_shortlist_preserves_bucket_coverage() -> None:
    records = []

    for bucket in ("a:addition", "b:division"):
        for index in range(5):
            records.append(
                screening_record(
                    example_id=f"{bucket}_{index}",
                    bucket=bucket,
                    status="wrong",
                    gold_log_probability=-float(index + 1),
                )
            )

    selected = select_shortlist(
        records,
        shortlist_size=6,
        minimum_per_bucket=2,
        maximum_per_bucket=4,
    )

    counts = Counter(
        record["bucket"]
        for record in selected
    )

    assert len(selected) == 6
    assert counts["a:addition"] >= 2
    assert counts["b:division"] >= 2
    assert max(counts.values()) <= 4


def test_discovery_uses_answer_only_gold_scoring() -> None:
    import inspect

    from usig.experiment.discover_reasoning_candidates import (
        score_gold_answer,
    )

    source = inspect.getsource(score_gold_answer)

    assert 'scoring_prefix = formatted_prompt + "FINAL: "' in source
    assert 'target_text = str(gold_answer)' in source
    assert '"gold_scoring_prefix": "FINAL: "' in source
