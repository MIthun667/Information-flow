from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from usig.data.ifi_arith import (
    BenchmarkBuild,
    materialize_benchmark,
    records_checksum,
    sample_pilot,
    validate_domain_balance,
)
from usig.data.loaders.common import SourceRecordError
from usig.data.loaders.ifi_arith import (
    compute_answer,
    normalize_ifi_arith_record,
    parse_arithmetic_question,
)
from usig.data.normalization.text import normalize_question
from usig.data.schema import CanonicalRecord
from usig.evaluation.arithmetic import (
    evaluate_arithmetic_answer,
    extract_final_integer_answer,
    extract_integer_answer,
)


def raw_record(
    operation: str = "addition",
    left: int = 3,
    right: int = 4,
    answer: str = "7",
) -> dict:
    symbol = {"addition": "+", "subtraction": "-", "multiplication": "×", "division": "÷"}.get(
        operation, "?"
    )
    return {
        "id": f"{operation}_0001",
        "task": operation,
        "operation": operation,
        "left_operand": left,
        "right_operand": right,
        "expression": f"{left} {symbol} {right}",
        "question": f"{left} {symbol} {right}",
        "prompt": f"{left} {symbol} {right} =",
        "reference_answer": answer,
    }


def canonical(
    *,
    domain: str = "source",
    seed: int = 2040,
    operation: str = "addition",
    index: int = 0,
) -> CanonicalRecord:
    operands = {
        "addition": (index + 10, 2),
        "subtraction": (index + 10, 2),
        "multiplication": (index + 10, 2),
        "division": ((index + 10) * 2, 2),
    }
    left, right = operands[operation]
    answer = compute_answer(operation, left, right)
    return normalize_ifi_arith_record(
        raw_record(operation, left, right, str(answer)),
        domain=domain,
        seed=seed,
        source_file="fixture.jsonl",
        record_index=index,
        operation_index=index,
    )


def test_source_record_normalizes_and_preserves_seed_domain_operation() -> None:
    record = canonical()
    assert record.dataset == "ifi_arith"
    assert record.task_family == "reasoning"
    assert record.metadata["seed"] == 2040
    assert record.domain == record.split == "source"
    assert record.category == "addition"


@pytest.mark.parametrize(
    ("operation", "left", "right", "answer"),
    [
        ("addition", 8, 4, 12),
        ("subtraction", 8, 4, 4),
        ("multiplication", 8, 4, 32),
        ("division", 8, 4, 2),
    ],
)
def test_each_operation_is_evaluated(
    operation: str, left: int, right: int, answer: int
) -> None:
    assert compute_answer(operation, left, right) == answer


def test_incorrect_stored_answer_is_detected() -> None:
    with pytest.raises(SourceRecordError, match="disagrees"):
        normalize_ifi_arith_record(
            raw_record(answer="8"),
            domain="source",
            seed=2040,
            source_file="fixture",
            record_index=0,
            operation_index=0,
        )


def test_non_exact_division_is_detected() -> None:
    with pytest.raises(SourceRecordError, match="not exact"):
        compute_answer("division", 7, 2)


def test_domain_mapping_and_deterministic_identifier() -> None:
    first = canonical(domain="larger_integer", seed=3040)
    second = canonical(domain="larger_integer", seed=3040)
    assert first.example_id == second.example_id
    assert first.example_id.startswith("ifi_arith:larger_integer:3040:addition:")


def test_identifiers_are_globally_unique_across_domain_seed_and_operation() -> None:
    records = [
        canonical(domain="source", seed=2040, operation="addition"),
        canonical(domain="source", seed=2041, operation="addition"),
        canonical(domain="larger_integer", seed=3040, operation="addition"),
        canonical(domain="source", seed=2040, operation="subtraction"),
    ]
    assert len({record.example_id for record in records}) == len(records)


def test_duplicate_questions_and_operand_tuples_are_detectable() -> None:
    records = [canonical(index=0), canonical(index=0)]
    assert len(records) - len({record.question for record in records}) == 1
    tuples = [
        (record.category, record.metadata["operand_a"], record.metadata["operand_b"])
        for record in records
    ]
    assert len(tuples) - len(set(tuples)) == 1


def test_cross_domain_normalized_duplicate_is_detectable() -> None:
    left = canonical(domain="source", seed=2040)
    right = canonical(domain="larger_integer", seed=3040)
    assert normalize_question(left.question) == normalize_question(right.question)


def test_unsupported_operation_fails() -> None:
    with pytest.raises(SourceRecordError, match="Unsupported"):
        compute_answer("power", 2, 3)


def test_missing_expected_answer_fails_with_context() -> None:
    record = raw_record()
    del record["reference_answer"]
    with pytest.raises(SourceRecordError, match="missing required field"):
        normalize_ifi_arith_record(
            record,
            domain="source",
            seed=2040,
            source_file="fixture",
            record_index=2,
            operation_index=0,
        )


def test_missing_operands_use_only_unambiguous_question() -> None:
    record = raw_record()
    del record["left_operand"]
    del record["right_operand"]
    normalized = normalize_ifi_arith_record(
        record,
        domain="source",
        seed=2040,
        source_file="fixture",
        record_index=0,
        operation_index=0,
    )
    assert normalized.metadata["operand_a"] == 3


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("3 + 4", (3, 4, "addition")),
        ("9 - 2", (9, 2, "subtraction")),
        ("7 × 8", (7, 8, "multiplication")),
        ("12 ÷ 3", (12, 3, "division")),
        ("-3 - -2", (-3, -2, "subtraction")),
    ],
)
def test_question_parser_handles_supported_formats(
    question: str, expected: tuple[int, int, str]
) -> None:
    assert parse_arithmetic_question(question) == expected


@pytest.mark.parametrize("question", ["3 plus 4", "3 + 4 = 7", "3 + 4 or 5", ""])
def test_question_parser_rejects_ambiguous_formats(question: str) -> None:
    with pytest.raises(SourceRecordError):
        parse_arithmetic_question(question)


def test_reference_answer_is_string_and_model_fields_are_absent() -> None:
    record = canonical()
    assert record.reference_answers == ["12"]
    forbidden = {"generated_answer", "correctness", "probability", "hidden_states", "ifi"}
    assert forbidden.isdisjoint(record.to_dict())


@pytest.mark.parametrize(
    ("domain", "seed", "operations"),
    [
        ("source", 2040, {"addition": 1, "subtraction": 1}),
        ("larger_integer", 3040, {"addition": 1, "division": 1}),
        ("moderate_multiplicative", 4040, {"multiplication": 1, "division": 1}),
    ],
)
def test_domain_balance_validation(
    domain: str, seed: int, operations: dict[str, int]
) -> None:
    records = [
        canonical(domain=domain, seed=seed, operation=operation, index=index)
        for operation, count in operations.items()
        for index in range(count)
    ]
    validate_domain_balance(
        domain,
        records,
        seeds=[seed],
        records_per_seed=sum(operations.values()),
        expected_operations=operations,
    )


def source_pilot_fixture() -> list[CanonicalRecord]:
    return [
        canonical(domain="source", seed=seed, operation=operation, index=index)
        for seed in range(2040, 2045)
        for operation in ("addition", "subtraction", "multiplication", "division")
        for index in range(5)
    ]


def test_pilot_is_balanced_deterministic_source_only_and_unique() -> None:
    records = source_pilot_fixture()
    first = sample_pilot(records)
    second = sample_pilot(records)
    assert first == second
    assert len(first) == 100
    assert len({item["example_id"] for item in first}) == 100
    assert {item["domain"] for item in first} == {"source"}
    counts: dict[str, int] = {}
    for item in first:
        operation = item["selection"]["operation"]
        counts[operation] = counts.get(operation, 0) + 1
    assert counts == {"addition": 25, "division": 25, "multiplication": 25, "subtraction": 25}


@pytest.mark.parametrize(
    ("prediction", "normalized"),
    [
        ("1,234", "1234"),
        ("-42", "-42"),
        ("The answer is 17.", "17"),
        ("8.0", "8"),
        ("  +9  ", "9"),
    ],
)
def test_answer_extraction_supported_cases(prediction: str, normalized: str) -> None:
    result = extract_integer_answer(prediction)
    assert result["parsing_status"] == "ok"
    assert result["normalized_answer"] == normalized


def test_multiple_numbers_fraction_and_nonnumeric_are_rejected() -> None:
    assert extract_integer_answer("2 plus 2 is 4")["error_reason"] == "ambiguous_multiple_numbers"
    assert extract_integer_answer("4/1")["error_reason"] == "fraction_not_allowed"
    assert extract_integer_answer("unknown")["error_reason"] == "no_numeric_answer"


def test_evaluator_returns_exact_match_and_status() -> None:
    result = evaluate_arithmetic_answer("Answer: 1,200", "1200")
    assert result["exact_match"] is True
    assert result["parsing_status"] == "ok"
    assert result["parsed_answer"] == "1,200"


def test_evaluator_prefers_explicit_final_answer() -> None:
    result = evaluate_arithmetic_answer(
        "There are 2 groups of 6, so the final answer is 12.", "12"
    )
    assert result["exact_match"] is True
    assert result["normalized_answer"] == "12"


def test_gsm_style_final_answer_parser_uses_last_or_boxed_number() -> None:
    assert extract_final_integer_answer("2 + 6 = 8. Therefore **8**.")[
        "normalized_answer"
    ] == "8"
    assert extract_final_integer_answer(r"work: 20 / 2 = \boxed{10}")[
        "normalized_answer"
    ] == "10"
    result = evaluate_arithmetic_answer(
        "There are 2 groups of 6, so 12.", "12", final_answer=True
    )
    assert result["exact_match"] is True


def test_inputs_are_not_mutated() -> None:
    source = raw_record()
    original = deepcopy(source)
    normalize_ifi_arith_record(
        source,
        domain="source",
        seed=2040,
        source_file="fixture",
        record_index=0,
        operation_index=0,
    )
    assert source == original


def test_repeated_records_have_identical_checksums() -> None:
    records = [canonical(index=index) for index in range(3)]
    assert records_checksum(records) == records_checksum(records)


def test_existing_outputs_are_protected(tmp_path: Path) -> None:
    target = tmp_path / "data" / "normalized" / "ifi_arith" / "source.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("existing\n", encoding="utf-8")
    records = source_pilot_fixture()
    build = BenchmarkBuild(
        records_by_domain={"source": records},
        audits={
            "checksums": {"source": records_checksum(records)},
            "summary": {"normalized_record_counts": {"source": len(records)}},
        },
    )
    description = tmp_path / "description.md"
    description.write_text("# Description\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materialize_benchmark(
            build,
            destination_root=tmp_path,
            include_pilot=False,
            overwrite=False,
            description_source=description,
        )
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_canonical_record_serialization_contains_no_parent_paths_by_import() -> None:
    payload = json.loads(canonical().to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["source"]["file"] == "fixture.jsonl"
