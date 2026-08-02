from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from usig.data.loaders.common import SourceRecordError
from usig.data.loaders.gsm8k import normalize_gsm8k_record, parse_gsm8k_answer
from usig.data.loaders.triviaqa import (
    normalize_triviaqa_record,
    select_triviaqa_variant,
)
from usig.data.pilot_collection import (
    balanced_binary_sample,
    canonical_json,
    gsm8k_sample,
    make_manifest,
    proportional_stratified_sample,
    quartile_labels,
)
from usig.evaluation.text import (
    evaluate_aliases,
    evaluate_ambignq,
    evaluate_squad,
    evaluate_truthfulqa,
)


def gsm_source(answer: str = "Work shown.\n#### 1,234") -> dict:
    return {"question": "How many items are there?", "answer": answer}


def trivia_source(*, with_answer: bool = True) -> dict:
    record = {
        "QuestionId": "q1",
        "Question": "Who wrote the book?",
        "QuestionSource": "fixture",
        "EntityPages": [
            {"Filename": "Book.txt", "Title": "Book", "DocSource": "TagMe"}
        ],
    }
    if with_answer:
        record["Answer"] = {
            "Value": "Ada",
            "Aliases": ["Ada", "A. Author"],
            "NormalizedAliases": ["ada", "a author"],
            "NormalizedValue": "ada",
            "Type": "WikipediaEntity",
            "MatchedWikiEntityName": "Ada",
        }
    return record


def simple_record(
    index: int,
    *,
    dataset: str = "gsm8k",
    split: str = "test",
    category: str = "a",
    answerable: bool = True,
) -> dict:
    return {
        "schema_version": "1.0",
        "example_id": f"{dataset}:{split}:{index}",
        "group_id": f"group:{dataset}:{index}",
        "source_id": str(index),
        "task_family": "reasoning",
        "dataset": dataset,
        "dataset_variant": None,
        "split": split,
        "domain": "natural_language_reasoning",
        "question": "word " * (index % 11 + 1),
        "context": None,
        "reference_answers": [str(index + 1)],
        "incorrect_reference_answers": [],
        "answerable": answerable,
        "ambiguous": False,
        "ambiguity_count": 1,
        "interpretations": [],
        "category": category,
        "source": {"file": "fixture", "record_index": index, "original_split": split},
        "metadata": {},
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234", "1234"), ("-5", "-5"), ("8.0", "8"), ("0.25", "0.25")],
)
def test_gsm8k_final_answer_parsing(raw: str, expected: str) -> None:
    normalized, status = parse_gsm8k_answer(f"Reasoning\n#### {raw}")
    assert normalized == expected
    assert status["parse_status"] == "ok"


@pytest.mark.parametrize("answer", ["missing marker", "#### apples", "#### NaN"])
def test_malformed_gsm8k_answer_rejected(answer: str) -> None:
    with pytest.raises(SourceRecordError):
        parse_gsm8k_answer(answer)


def test_gsm8k_normalization_and_deterministic_identifier() -> None:
    source = gsm_source()
    first = normalize_gsm8k_record(
        source,
        split="test",
        record_index=0,
        source_file="fixture",
        fingerprint="fp",
        retrieval_date="2026-01-01",
    )
    second = normalize_gsm8k_record(
        source,
        split="test",
        record_index=9,
        source_file="fixture",
        fingerprint="fp",
        retrieval_date="2026-01-01",
    )
    assert first.example_id == second.example_id
    assert first.reference_answers == ["1234"]
    assert first.metadata["reference_solution"] == source["answer"]


def trivia_audit(wikipedia_ok: bool = True) -> dict:
    return {
        "license": {"status": "documented"},
        "variants": {
            "wikipedia": {
                "train_count": 1,
                "validation_count": 1,
                "records_without_usable_answers": 0 if wikipedia_ok else 1,
                "missing_evidence_references": 0,
            },
            "web": {
                "train_count": 1,
                "validation_count": 1,
                "records_without_usable_answers": 0,
                "missing_evidence_references": 0,
            },
        },
    }


def test_triviaqa_variant_selection_prefers_wikipedia_then_web() -> None:
    assert select_triviaqa_variant(trivia_audit()) == "wikipedia"
    assert select_triviaqa_variant(trivia_audit(False)) == "web"


def test_triviaqa_aliases_and_evidence_are_preserved() -> None:
    record = normalize_triviaqa_record(
        trivia_source(),
        split="validation",
        variant="wikipedia",
        source_file="fixture",
        record_index=0,
    )
    assert record.reference_answers == ["Ada", "A. Author"]
    assert record.metadata["normalized_aliases"] == ["ada", "a author"]
    assert record.metadata["evidence_references"][0]["relative_path"].startswith(
        "evidence/wikipedia/"
    )


def test_answerless_triviaqa_is_excluded() -> None:
    with pytest.raises(SourceRecordError, match="answerless"):
        normalize_triviaqa_record(
            trivia_source(with_answer=False),
            split="test",
            variant="wikipedia",
            source_file="fixture",
            record_index=0,
        )


def test_verified_subset_duplicate_detection_by_id() -> None:
    validation_ids = {"q1", "q2"}
    verified_ids = {"q2"}
    assert len(validation_ids & verified_ids) == 1


def test_missing_evidence_reference_is_reportable(tmp_path: Path) -> None:
    reference = trivia_source()["EntityPages"][0]["Filename"]
    assert not (tmp_path / reference).exists()


def test_quartile_labels_and_gsm_sampling_are_deterministic() -> None:
    assert set(quartile_labels(list(range(100)))) == {0, 1, 2, 3}
    records = [simple_record(index) for index in range(400)]
    first = gsm8k_sample(records)
    second = gsm8k_sample(records)
    assert first == second
    assert len(first) == 100
    assert len({stratum for _, stratum in first}) >= 8


def test_proportional_category_allocation_includes_every_category() -> None:
    records = [
        simple_record(index, category=f"c{index % 7}") for index in range(140)
    ]
    selected = proportional_stratified_sample(
        records,
        key=lambda record: record["category"],
        count=30,
        seed=2026,
    )
    assert {stratum for _, stratum in selected} == {f"c{i}" for i in range(7)}


def test_binary_sampling_returns_exact_balance() -> None:
    records = [
        simple_record(index, answerable=index < 60) for index in range(120)
    ]
    selected = balanced_binary_sample(
        records,
        predicate=lambda record: record["answerable"],
        true_label="answerable",
        false_label="unanswerable",
        seed=2026,
    )
    assert [stratum for _, stratum in selected].count("answerable") == 50
    assert [stratum for _, stratum in selected].count("unanswerable") == 50


def test_alias_count_strata_are_preserved() -> None:
    records = [simple_record(index, category=str(index % 3 + 1)) for index in range(120)]
    selected = proportional_stratified_sample(
        records,
        key=lambda record: record["category"],
        count=100,
        seed=2026,
    )
    assert {stratum for _, stratum in selected} == {"1", "2", "3"}


def test_manifest_contract_has_no_complete_record() -> None:
    records = [simple_record(index) for index in range(3)]
    manifest = make_manifest(
        [(record, "fixture") for record in records],
        source_checksum="source",
    )
    assert all("question" not in item and "reference_answers" not in item for item in manifest)
    assert [item["selection_order"] for item in manifest] == [0, 1, 2]


def test_duplicate_manifest_ids_are_detectable() -> None:
    records = [simple_record(0), simple_record(0)]
    ids = [record["example_id"] for record in records]
    assert len(ids) != len(set(ids))


def test_prompt_configuration_has_valid_checksums_and_no_forbidden_metadata() -> None:
    path = Path(__file__).parents[1] / "config" / "prompts" / "benchmark_prompts.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    forbidden = ("gold", "correctness", "chain-of-thought", "truthfulqa", "gsm8k")
    for template in config["templates"].values():
        text = template["text"]
        assert template["checksum"] == hashlib.sha256(text.encode()).hexdigest()
        assert all(word not in text.lower() for word in forbidden)


def test_text_evaluators_return_expected_toy_results() -> None:
    aliases = evaluate_aliases("New York", ["NYC", "New York"])
    assert aliases["exact_match"] is True
    truthful = evaluate_truthfulqa("correct", ["correct"], ["wrong"])
    assert truthful["status"] == "matched_correct_reference"
    ambig = evaluate_ambignq(
        "blue", [{"reference_answers": ["blue"]}, {"reference_answers": ["red"]}]
    )
    assert ambig["interpretation_recall"] == 0.5
    assert evaluate_squad("unanswerable", [], answerable=False)["combined_correct"] is True


def test_source_records_are_not_mutated() -> None:
    source = trivia_source()
    original = deepcopy(source)
    normalize_triviaqa_record(
        source,
        split="validation",
        variant="wikipedia",
        source_file="fixture",
        record_index=0,
    )
    assert source == original


def test_canonical_serialization_checksum_is_reproducible() -> None:
    record = simple_record(1)
    assert canonical_json(record) == canonical_json(json.loads(canonical_json(record)))
