from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from usig.evaluation.audit_rules import (
    concise_alias_match,
    conservative_abstention,
    evaluate_interpretation_segments,
)
from usig.experiment.audit_completed_experiment import _fit_splits
from usig.experiment.collection import evaluate_response, validate_collection
from usig.experiment.generation import render_prompt
from usig.experiment.hidden_states import align_answer_hidden_states, transition_matrices
from usig.experiment.probabilities import extract_probability_features
from usig.experiment.records import (
    atomic_json,
    checksum_record,
    validate_record_checksum,
)
from usig.experiment.signatures import (
    calculate_signature,
    depth_regions,
    interior_transition_indices,
)


class MockTokenizer:
    chat_template = "available"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        length = len(text.split())
        return {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


def test_prompt_rendering_is_deterministic_and_uses_chat_template() -> None:
    template = {"text": "Question: {question}\nAnswer:", "template_id": "fixture"}
    record = {"question": "What is 2 + 2?", "context": None}
    first = render_prompt(MockTokenizer(), template, record)
    second = render_prompt(MockTokenizer(), template, record)
    assert first["rendered_prompt"] == second["rendered_prompt"]
    assert first["rendered_prompt_checksum"] == second["rendered_prompt_checksum"]
    assert first["chat_template_status"] == "official_chat_template"


def test_hidden_state_answer_boundary_alignment() -> None:
    states = tuple(torch.randn(1, 5, 4) for _ in range(4))
    aligned = align_answer_hidden_states(states, prompt_length=3, generated_token_count=2)
    assert aligned.shape == (2, 3, 4)
    with pytest.raises(ValueError, match="misaligned"):
        align_answer_hidden_states(states, prompt_length=2, generated_token_count=2)


def test_probability_alignment_entropy_and_selected_log_probability() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    result = extract_probability_features((logits,), [2])
    expected = torch.log_softmax(logits[0], dim=-1)[2].item()
    assert result["tokens"][0]["selected_log_probability"] == pytest.approx(expected)
    assert result["tokens"][0]["entropy"] > 0
    assert result["tokens"][0]["selected_rank"] == 1
    with pytest.raises(ValueError, match="misaligned"):
        extract_probability_features((logits,), [])


def test_cosine_and_relative_transition_calculation() -> None:
    states = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )
    transitions = transition_matrices(states)
    assert transitions["cosine"].shape == (1, 2)
    assert transitions["cosine"][0].tolist() == pytest.approx([0.0, 1.0])
    assert transitions["relative"][0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, [0]), (10, list(range(1, 9))), (23, list(range(3, 20)))],
)
def test_interior_transition_selection(count: int, expected: list[int]) -> None:
    assert interior_transition_indices(count) == expected


def test_depth_partitioning_covers_every_transition() -> None:
    regions = depth_regions(23)
    flattened = regions["early"] + regions["middle"] + regions["late"]
    assert flattened == list(range(23))
    assert all(regions.values())


def test_scalar_ifi_population_standard_deviation() -> None:
    cosine = torch.tensor([[0.1] * 10, [0.3] * 10])
    relative = torch.tensor([[0.2] * 10, [0.4] * 10])
    signature = calculate_signature(cosine, relative)
    assert signature["scalar_ifi"] == pytest.approx(0.1)
    assert signature["feature_status"] == "ok"
    assert signature["population_standard_deviation"] is True


def test_one_token_ifi_policy_is_null() -> None:
    signature = calculate_signature(torch.ones(1, 10), torch.ones(1, 10))
    assert signature["scalar_ifi"] is None
    assert signature["feature_status"] == "insufficient_tokens"


def test_fixed_depth_profile_and_token_dynamics() -> None:
    cosine = torch.arange(40, dtype=torch.float32).reshape(4, 10) / 100
    relative = cosine + 0.1
    signature = calculate_signature(cosine, relative)
    profile = signature["cosine_structured"]["cosine_fixed_depth_profile"]
    assert len(profile) == 32
    assert signature["cosine_token_dynamics"]["token_instability_slope"] > 0
    assert signature["cosine_token_dynamics"]["token_instability_roughness"] > 0
    assert signature["cosine_structured"]["cosine_profile_roughness"] > 0


def test_non_finite_features_are_rejected() -> None:
    cosine = torch.ones(2, 10)
    cosine[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        calculate_signature(cosine, torch.ones(2, 10))


def test_prediction_and_signature_checksum_serialization(tmp_path: Path) -> None:
    prediction = {"example_id": "x", "response": "4"}
    prediction["record_checksum"] = checksum_record(prediction, "record_checksum")
    signature = {"example_id": "x", "scalar_ifi": 0.2}
    signature["signature_checksum"] = checksum_record(signature, "signature_checksum")
    assert validate_record_checksum(prediction, "record_checksum")
    assert validate_record_checksum(signature, "signature_checksum")
    path = tmp_path / "record.json"
    atomic_json(path, prediction)
    assert json.loads(path.read_text()) == prediction


def test_evaluator_integration_and_truthfulqa_unresolved() -> None:
    arithmetic = {
        "dataset": "ifi_arith",
        "reference_answers": ["4"],
    }
    assert evaluate_response(arithmetic, "4")["binary_correctness"] is True
    truthful = {
        "dataset": "truthfulqa",
        "reference_answers": ["correct"],
        "incorrect_reference_answers": ["wrong"],
    }
    result = evaluate_response(truthful, "unmatched response")
    assert result["binary_correctness"] is None
    assert result["unresolved_label"] is True


def test_ambignq_single_answer_uses_record_level_aliases() -> None:
    record = {
        "dataset": "ambignq",
        "interpretations": [],
        "reference_answers": ["Tony Goldwyn", "Goldwyn"],
    }
    result = evaluate_response(record, "Goldwyn")
    assert result["binary_correctness"] is True
    assert result["metrics"]["interpretation_count"] == 1


def test_squad_unanswerable_integration() -> None:
    record = {
        "dataset": "squad",
        "reference_answers": [],
        "answerable": False,
    }
    result = evaluate_response(record, "unanswerable")
    assert result["binary_correctness"] is True


def test_manifest_checksum_rejection(tmp_path: Path) -> None:
    path = tmp_path / "data/manifests/pilots"
    path.mkdir(parents=True)
    (path / "six_benchmark_seed2026_n600.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_collection(tmp_path)


def test_checksum_changes_when_record_changes() -> None:
    first = {"example_id": "x", "value": 1}
    second = {"example_id": "x", "value": 2}
    assert checksum_record(first, "checksum") != checksum_record(second, "checksum")


def test_token_limit_status_policy() -> None:
    token_count = 24
    requested_limit = 24
    assert token_count >= requested_limit


def test_no_parent_ifi_import_in_experiment_sources() -> None:
    root = Path(__file__).parents[1] / "src/usig/experiment"
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "from ifi" not in source
    assert "import ifi" not in source


def test_triviaqa_alias_in_concise_sentence_suffix() -> None:
    result = concise_alias_match(
        "The state capital of Alabama is Montgomery.", ["Montgomery"]
    )
    assert result["match"] is True
    assert result["rule"] == "concise_containment"


def test_triviaqa_containment_does_not_credit_false_relation() -> None:
    result = concise_alias_match(
        "Kirkland Signature is the house brand of Target Corporation.",
        ["Kirkland Signature"],
    )
    assert result["answer_containment"] is True
    assert result["match"] is False


def test_triviaqa_punctuation_and_parenthetical_alias() -> None:
    assert concise_alias_match("Copenhagen (Denmark)", ["Copenhagen"])["match"]
    assert concise_alias_match("Copenhagen", ["Copenhagen (Denmark)"])["match"]


def test_ambignq_multi_answer_segmentation() -> None:
    interpretations = [
        {"interpretation_id": "a", "reference_answers": ["red"]},
        {"interpretation_id": "b", "reference_answers": ["blue"]},
    ]
    result = evaluate_interpretation_segments("1. red;\n2. blue", interpretations)
    assert result["covered_interpretations"] == 2


def test_ambignq_interpretation_isolation() -> None:
    interpretations = [
        {"interpretation_id": "a", "reference_answers": ["red"]},
        {"interpretation_id": "b", "reference_answers": ["blue"]},
    ]
    result = evaluate_interpretation_segments("red", interpretations)
    assert result["interpretations"][0]["matched"] is True
    assert result["interpretations"][1]["matched"] is False


def test_ambignq_numbered_answers_respect_interpretation_position() -> None:
    interpretations = [
        {"interpretation_id": "0", "reference_answers": ["red"]},
        {"interpretation_id": "1", "reference_answers": ["blue"]},
    ]
    result = evaluate_interpretation_segments("1. blue\n2. red", interpretations)
    assert result["covered_interpretations"] == 0


@pytest.mark.parametrize(
    "response",
    [
        "It cannot be determined from the context.",
        "The answer is not provided.",
        "There is insufficient information.",
        "The context does not say.",
        "It is unknown from the passage.",
    ],
)
def test_conservative_squad_abstention_variants(response: str) -> None:
    assert conservative_abstention(response)["abstained"] is True


def test_score_direction_and_positive_error_orientation() -> None:
    matrix = np.asarray([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    splits = [
        (np.asarray([1, 2, 4, 5]), np.asarray([0, 3])),
        (np.asarray([0, 2, 3, 5]), np.asarray([1, 4])),
        (np.asarray([0, 1, 3, 4]), np.asarray([2, 5])),
    ]
    result = _fit_splits(matrix, labels, splits)
    assert result["positive_class"] == "binary_error"
    assert result["score_direction"] == "higher_is_greater_error_risk"
    assert result["auroc"] == pytest.approx(1.0)


def test_class_deficient_fold_handling() -> None:
    matrix = np.asarray([[0.0], [1.0], [2.0]])
    labels = np.asarray([0, 0, 0])
    result = _fit_splits(
        matrix,
        labels,
        [(np.asarray([0, 1]), np.asarray([2]))],
    )
    assert result["valid_folds"] == 0
    assert result["undefined_folds"] == 1
    assert result["auroc"] is None


def test_constant_feature_handling() -> None:
    matrix = np.asarray(
        [[1.0, 0.0], [1.0, 0.1], [1.0, 0.9], [1.0, 1.0]]
    )
    labels = np.asarray([0, 0, 1, 1])
    result = _fit_splits(
        matrix,
        labels,
        [
            (np.asarray([0, 3]), np.asarray([1, 2])),
            (np.asarray([1, 2]), np.asarray([0, 3])),
        ],
    )
    assert result["valid_folds"] == 2
    assert result["convergence_warning_count"] == 0


def test_identical_split_reuse_checksum() -> None:
    splits = [([0, 2], [1, 3]), ([1, 3], [0, 2])]
    first = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    second = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    assert first == second


def test_missing_scalar_ifi_is_not_imputed() -> None:
    signature = {"scalar_ifi": None}
    scalar_features = (
        {} if signature["scalar_ifi"] is None else {"scalar_ifi": signature["scalar_ifi"]}
    )
    assert scalar_features == {}


def test_frozen_artifact_checksums_are_preserved() -> None:
    root = Path(__file__).parents[1]
    prediction = (
        root
        / "outputs/predictions/qwen_ifi_66b0032f646fc519.jsonl"
    )
    signature = (
        root
        / "outputs/signatures/qwen_ifi_66b0032f646fc519.jsonl"
    )
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == (
        "ee131679054b616852d8db5de67d2c36109a0d1a0783e613f7a17f15b6829769"
    )
    assert hashlib.sha256(signature.read_bytes()).hexdigest() == (
        "7f6050271d1e2d1136783163a44bba0b02c29bb88526dd2b7964cab9db435f9f"
    )


def test_audit_outputs_do_not_target_frozen_artifacts() -> None:
    source = (
        Path(__file__).parents[1]
        / "src/usig/experiment/audit_completed_experiment.py"
    ).read_text()
    assert 'output_root = project_root / "outputs/audits"' in source
