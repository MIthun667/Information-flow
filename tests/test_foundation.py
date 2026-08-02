from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from usig.data.identifiers import example_identifier, truthfulqa_source_id
from usig.data.leakage import LeakageRegistry
from usig.data.loaders.ambignq import normalize_ambignq_record
from usig.data.loaders.common import SourceRecordError
from usig.data.loaders.nqopen import normalize_nqopen_record
from usig.data.loaders.squad import normalize_squad_record
from usig.data.loaders.truthfulqa import normalize_truthfulqa_record
from usig.data.normalization.text import normalized_question_hash
from usig.data.prepare import _atomic_jsonl, load_configs, prepare
from usig.data.schema import CanonicalRecord
from usig.data.triviaqa import inspect_triviaqa


def single_ambignq() -> dict:
    return {
        "id": "10",
        "question": "Who wrote it?",
        "annotations": [{"type": "singleAnswer", "answer": ["A. Writer", ""]}],
    }


def multiple_ambignq() -> dict:
    return {
        "id": "11",
        "question": "When did it begin?",
        "annotations": [
            {
                "type": "multipleQAs",
                "qaPairs": [
                    {"question": "When did event A begin?", "answer": ["1900"]},
                    {"question": "When did event B begin?", "answer": ["2000"]},
                ],
            }
        ],
    }


def nq_record(source_id: str = "10", question: str = "who wrote it") -> dict:
    return {"id": source_id, "question": question, "answer": ["A. Writer", "Writer"]}


def truthfulqa_record() -> dict:
    return {
        "Type": "Adversarial",
        "Category": "Test",
        "Question": "Is this true?",
        "Best Answer": "Yes",
        "Best Incorrect Answer": "No",
        "Correct Answers": "Yes; Correct",
        "Incorrect Answers": "No; Incorrect",
        "Source": "fixture",
    }


def squad_record(answerable: bool = True) -> dict:
    context = "The sky is blue."
    return {
        "id": "s1",
        "title": "Sky",
        "context": context,
        "question": "What color is the sky?",
        "answers": {"text": ["blue"] if answerable else [], "answer_start": [11] if answerable else []},
    }


def test_canonical_record_serializes_and_removes_empty_aliases() -> None:
    record = normalize_ambignq_record(
        single_ambignq(), split="validation", source_file="fixture", record_index=0
    )
    assert json.loads(record.to_json())["example_id"] == "ambignq:validation:10"
    assert record.reference_answers == ["A. Writer"]


def test_identifiers_and_question_hashes_are_deterministic() -> None:
    assert example_identifier("nqopen", "test", "12") == example_identifier(
        "nqopen", "test", "12"
    )
    assert normalized_question_hash("Hello, World!") == normalized_question_hash("hello world")


def test_ambignq_single_answer_and_input_immutability() -> None:
    source = single_ambignq()
    original = deepcopy(source)
    record = normalize_ambignq_record(
        source, split="train", source_file="fixture", record_index=0
    )
    assert record.ambiguous is False and record.ambiguity_count == 1
    assert source == original


def test_ambignq_interpretations_remain_grouped() -> None:
    record = normalize_ambignq_record(
        multiple_ambignq(), split="validation", source_file="fixture", record_index=0
    )
    assert record.ambiguous is True
    assert record.ambiguity_count == 2
    assert [item.reference_answers for item in record.interpretations] == [["1900"], ["2000"]]


def test_nqopen_aliases_and_null_answerability() -> None:
    record = normalize_nqopen_record(
        nq_record(), split="test", source_file="fixture", record_index=0
    )
    assert record.reference_answers == ["A. Writer", "Writer"]
    assert record.answerable is None


def test_truthfulqa_references_are_separate_and_ids_stable() -> None:
    first = normalize_truthfulqa_record(
        truthfulqa_record(), source_file="fixture", record_index=0
    )
    second = normalize_truthfulqa_record(
        truthfulqa_record(), source_file="fixture", record_index=9
    )
    assert first.source_id == truthfulqa_source_id(first.question) == second.source_id
    assert first.reference_answers == ["Yes", "Correct"]
    assert first.incorrect_reference_answers == ["No", "Incorrect"]


def test_squad_answerable_span_and_unanswerable_record() -> None:
    answerable = normalize_squad_record(
        squad_record(), split="validation", source_file="fixture", record_index=0
    )
    unanswerable = normalize_squad_record(
        {**squad_record(False), "id": "s2"},
        split="validation",
        source_file="fixture",
        record_index=1,
    )
    assert answerable.answerable is True and answerable.reference_answers == ["blue"]
    assert unanswerable.answerable is False and unanswerable.reference_answers == []


def test_invalid_squad_span_fails_with_source_context() -> None:
    source = squad_record()
    source["answers"]["answer_start"] = [0]
    with pytest.raises(SourceRecordError, match="fixture record 4"):
        normalize_squad_record(
            source, split="validation", source_file="fixture", record_index=4
        )


def test_missing_required_field_fails_with_source_context() -> None:
    source = single_ambignq()
    del source["question"]
    with pytest.raises(SourceRecordError, match="fixture record 3"):
        normalize_ambignq_record(
            source, split="train", source_file="fixture", record_index=3
        )


def test_duplicate_example_id_is_detected() -> None:
    record = normalize_nqopen_record(
        nq_record(), split="train", source_file="fixture", record_index=0
    )
    registry = LeakageRegistry()
    registry.add(record)
    with pytest.raises(ValueError, match="Duplicate example ID"):
        registry.add(record)


def test_shared_source_and_normalized_collision_are_detected() -> None:
    ambig = normalize_ambignq_record(
        single_ambignq(), split="validation", source_file="ambig", record_index=0
    )
    shared = normalize_nqopen_record(
        nq_record(), split="train", source_file="nq", record_index=0
    )
    collision = normalize_nqopen_record(
        nq_record("99", "Who wrote it"), split="test", source_file="nq", record_index=1
    )
    registry = LeakageRegistry()
    registry.extend([ambig, shared, collision])
    overlaps = registry.overlaps()
    reasons = {item.reason for item in overlaps}
    assert "shared_source_id" in reasons
    assert "normalized_question_overlap" in reasons
    assert "cross_split_overlap" in reasons
    assert all(item.excluded_example_id and item.matched_example_id for item in overlaps)


def test_repeated_normalization_is_deterministic() -> None:
    first = normalize_nqopen_record(
        nq_record(), split="train", source_file="fixture", record_index=0
    )
    second = normalize_nqopen_record(
        nq_record(), split="train", source_file="fixture", record_index=0
    )
    assert first.to_json() == second.to_json()


def test_existing_output_is_protected(tmp_path: Path) -> None:
    destination = tmp_path / "records.jsonl"
    destination.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _atomic_jsonl(destination, [{"new": True}], overwrite=False)
    assert destination.read_text(encoding="utf-8") == "existing\n"


def test_triviaqa_evidence_is_not_interpreted_as_metadata(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence" / "web"
    evidence.mkdir(parents=True)
    (evidence / "document.txt").write_text("Evidence", encoding="utf-8")
    status = inspect_triviaqa(tmp_path)
    assert status["status"] == "pending"
    assert status["evidence_document_count"] == 1
    assert status["evidence_documents_are_examples"] is False


def test_example_ids_are_namespace_qualified() -> None:
    assert example_identifier("ambignq", "validation", "42").split(":") == [
        "ambignq",
        "validation",
        "42",
    ]


def test_normalized_duplicate_question_is_detected() -> None:
    first = normalize_nqopen_record(
        nq_record("1", "Who wrote it?"), split="train", source_file="a", record_index=0
    )
    second = normalize_nqopen_record(
        nq_record("2", "who wrote it"), split="train", source_file="b", record_index=0
    )
    registry = LeakageRegistry()
    registry.extend([first, second])
    assert "normalized_question_overlap" in registry.summary()


def test_cross_split_overlap_is_reported() -> None:
    first = normalize_nqopen_record(
        nq_record("1", "Same question"), split="train", source_file="a", record_index=0
    )
    second = normalize_nqopen_record(
        nq_record("2", "Same question"), split="validation", source_file="b", record_index=0
    )
    registry = LeakageRegistry()
    registry.extend([first, second])
    assert registry.summary()["cross_split_overlap"] >= 1


def test_overlap_preserves_both_identifiers() -> None:
    first = normalize_nqopen_record(
        nq_record("1", "Same question"), split="train", source_file="a", record_index=0
    )
    second = normalize_nqopen_record(
        nq_record("2", "Same question"), split="test", source_file="b", record_index=0
    )
    registry = LeakageRegistry()
    registry.extend([first, second])
    overlap = registry.overlaps()[0]
    assert {overlap.excluded_example_id, overlap.matched_example_id} == {
        first.example_id,
        second.example_id,
    }


def test_empty_nqopen_aliases_fail() -> None:
    with pytest.raises(SourceRecordError, match="no usable answer aliases"):
        normalize_nqopen_record(
            {"id": "1", "question": "Question?", "answer": ["", " "]},
            split="train",
            source_file="fixture",
            record_index=0,
        )


def test_answerability_is_null_when_source_does_not_define_it() -> None:
    assert (
        normalize_ambignq_record(
            single_ambignq(), split="train", source_file="fixture", record_index=0
        ).answerable
        is None
    )


def test_ambiguity_is_null_when_source_does_not_define_it() -> None:
    assert (
        normalize_nqopen_record(
            nq_record(), split="train", source_file="fixture", record_index=0
        ).ambiguous
        is None
    )


def test_generated_fields_are_absent_from_canonical_schema() -> None:
    fields = CanonicalRecord.__dataclass_fields__
    forbidden = {"generated_answer", "correctness", "model_score", "ifi"}
    assert forbidden.isdisjoint(fields)


def test_dataset_configurations_include_selected_triviaqa() -> None:
    configs = load_configs(Path(__file__).parents[1] / "config" / "datasets")
    assert configs["triviaqa"]["enabled"] is True
    assert configs["triviaqa"]["selected_variant"] == "wikipedia"
    assert configs["squad"]["raw_source"] == {
        "dataset_identifier": "squad_v2",
        "configuration_name": "squad_v2",
    }


def test_pending_triviaqa_does_not_block_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import usig.data.prepare as preparation

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "triviaqa.yaml").write_text(
        "dataset: triviaqa\nenabled: false\n", encoding="utf-8"
    )
    (config_dir / "truthfulqa.yaml").write_text(
        """dataset: truthfulqa
enabled: true
raw_source:
  all: data/raw/TruthfulQA.csv
normalized_destination: data/normalized/truthfulqa
""",
        encoding="utf-8",
    )
    record = normalize_truthfulqa_record(
        truthfulqa_record(), source_file="fixture", record_index=0
    )
    monkeypatch.setattr(preparation, "collect_dataset", lambda config: ({"all": [record]}, {}))
    result = prepare(["all"], config_dir=config_dir, audit_only=True)
    assert result["datasets"]["truthfulqa"]["normalized_count"] == 1


def test_audit_only_does_not_write_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import usig.data.prepare as preparation

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "nqopen.yaml").write_text(
        """dataset: nqopen
enabled: true
raw_source: {}
normalized_destination: data/normalized/nqopen
""",
        encoding="utf-8",
    )
    record = normalize_nqopen_record(
        nq_record(), split="train", source_file="fixture", record_index=0
    )
    monkeypatch.setattr(preparation, "collect_dataset", lambda config: ({"train": [record]}, {}))
    monkeypatch.setattr(preparation, "PROJECT_ROOT", tmp_path)
    prepare(["all"], config_dir=config_dir, audit_only=True)
    assert not (tmp_path / "data" / "normalized").exists()
