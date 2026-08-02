from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from usig.experiment.hidden_states import transition_matrices
from usig.experiment.large_collection import (
    _load_model,
    compact_features,
    secondary_features,
)
from usig.experiment.records import (
    atomic_json,
    canonical_json,
    checksum_record,
    compile_jsonl,
    read_valid_records,
    validate_record_checksum,
)
from usig.experiment.repair_v3 import read_jsonl
from usig.experiment.signatures import calculate_signature

VERSION = "truthfulqa_mc_v5"


def ordered_options(record: dict[str, Any]) -> list[dict[str, Any]]:
    options = [
        *[
            {
                "text": value,
                "correct": True,
                "mc1_correct": value == record["metadata"]["best_answer"],
            }
            for value in record["reference_answers"]
        ],
        *[
            {"text": value, "correct": False, "mc1_correct": False}
            for value in record["incorrect_reference_answers"]
        ],
    ]
    seed = int(hashlib.sha256(record["example_id"].encode()).hexdigest()[:16], 16)
    return sorted(
        options,
        key=lambda option: hashlib.sha256(
            f"{seed}:{option['text']}".encode()
        ).hexdigest(),
    )


def option_probabilities(total_log_probabilities: list[float]) -> list[float]:
    values = torch.tensor(total_log_probabilities, dtype=torch.float64)
    return torch.softmax(values, dim=0).tolist()


def score_option(
    model: Any, tokenizer: Any, prompt: str, option: str
) -> dict[str, Any]:
    if not option.strip():
        raise ValueError("TruthfulQA option is empty after normalization")
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    # Tokenize the answer separately and concatenate IDs. Tokenizing the
    # combined text is unsafe at the boundary: a trailing prompt-space token
    # can merge with a one-token option and make the apparent continuation
    # length zero.
    option_ids = tokenizer(
        option, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    if option_ids.shape[1] == 0:
        raise ValueError("TruthfulQA option produced no continuation tokens")
    full_ids = torch.cat((prompt_ids, option_ids), dim=1)
    device = next(model.parameters()).device
    ids = full_ids.to(device)
    with torch.inference_mode():
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    continuation = ids[0, prompt_length:]
    logits = output.logits[0, prompt_length - 1 : -1].float()
    selected = torch.log_softmax(logits, dim=-1).gather(
        1, continuation[:, None]
    )[:, 0]
    answer_hidden = torch.stack(
        [
            state[0, prompt_length:, :].detach().float().cpu()
            for state in output.hidden_states[1:]
        ],
        dim=1,
    )
    transitions = transition_matrices(answer_hidden)
    signature = calculate_signature(
        transitions["cosine"], transitions["relative"]
    )
    return {
        "token_count": int(len(continuation)),
        "total_log_probability": float(selected.sum()),
        "mean_log_probability": float(selected.mean()),
        "minimum_log_probability": float(selected.min()),
        "log_probability_std": float(selected.std(unbiased=False)),
        "signature": signature,
    }


def _probability_summary(option_scores: list[dict[str, Any]]) -> dict[str, float]:
    totals = [item["total_log_probability"] for item in option_scores]
    probabilities = option_probabilities(totals)
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    selected = max(option_scores, key=lambda item: item["total_log_probability"])
    return {
        "mean_token_entropy": entropy,
        "maximum_token_entropy": entropy,
        "negative_mean_log_probability": -selected["mean_log_probability"],
        "minimum_selected_token_log_probability": selected[
            "minimum_log_probability"
        ],
        "selected_token_log_probability_std": selected[
            "log_probability_std"
        ],
    }


def collect(
    model_identifier: str,
    manifest_path: Path,
    normalized_path: Path,
    destination: Path,
) -> dict[str, Any]:
    completed_report = destination / "verification_reports/artifact_checksums.json"
    if completed_report.exists():
        existing = verify(manifest_path, destination)
        if existing["complete"]:
            return {"already_complete": True, **existing}
        raise FileExistsError(
            f"Completed report exists but collection is invalid: {completed_report}"
        )
    manifest = read_jsonl(manifest_path)
    canonical = {
        item["example_id"]: item for item in read_jsonl(normalized_path)
    }
    model, tokenizer, metadata = _load_model(model_identifier)
    experiment_id = VERSION + "_" + hashlib.sha256(
        canonical_json(
            {
                "model": model_identifier,
                "revision": metadata["model_revision"],
                "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        ).encode()
    ).hexdigest()[:16]
    prediction_directory = destination / "predictions/records"
    compact_directory = destination / "compact_signatures/records"
    ablation_directory = destination / "signature_ablations/records"
    valid_predictions = read_valid_records(prediction_directory, "record_checksum")
    valid_compact = read_valid_records(compact_directory, "signature_checksum")
    valid_ablation = read_valid_records(ablation_directory, "ablation_checksum")
    metadata_path = destination / "extraction_metadata/experiment.json"
    metadata_value = {
            "version": VERSION,
            "experiment_id": experiment_id,
            "model": metadata,
            "manifest": str(manifest_path),
            "protocol": "TruthfulQA MC1 best-correct versus incorrect options",
            "generative_lexical_labels_are_primary": False,
        }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata_value:
            raise FileExistsError(f"Conflicting experiment metadata: {metadata_path}")
    else:
        atomic_json(metadata_path, metadata_value)
    for item in manifest:
        identifier = item["example_id"]
        if identifier in valid_predictions and identifier in valid_compact:
            continue
        record = canonical[identifier]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        record_start = time.perf_counter()
        options = ordered_options(record)
        checksum = hashlib.sha256(
            canonical_json([option["text"] for option in options]).encode()
        ).hexdigest()
        if checksum != item["option_order_checksum"]:
            raise ValueError(f"Option order checksum mismatch: {identifier}")
        rendered = (
            "Select the single best answer to the question.\n\n"
            f"Question: {record['question']}\nAnswer: "
        )
        scores = [
            score_option(model, tokenizer, rendered, option["text"])
            for option in options
        ]
        probabilities = option_probabilities(
            [score["total_log_probability"] for score in scores]
        )
        predicted = max(
            range(len(scores)),
            key=lambda index: scores[index]["total_log_probability"],
        )
        selected_signature = scores[predicted]["signature"]
        prediction = {
            "version": VERSION,
            "experiment_id": experiment_id,
            "example_id": identifier,
            "group_id": record["group_id"],
            "dataset": "truthfulqa_mc",
            "domain": "all",
            "model_identifier": model_identifier,
            "model_revision": metadata["model_revision"],
            "prompt_token_count": len(
                tokenizer(rendered, add_special_tokens=False)["input_ids"]
            ),
            "generated_token_count": scores[predicted]["token_count"],
            "response_character_count": len(options[predicted]["text"]),
            "token_limit_reached": False,
            "predicted_option_index": predicted,
            "correct_option_index": item["correct_option_index"],
            "correct_option_indices": item["correct_option_indices"],
            "option_probabilities": probabilities,
            "binary_correctness": predicted == item["correct_option_index"],
            "binary_error": int(predicted != item["correct_option_index"]),
            "unresolved_label": False,
            "total_runtime_seconds": time.perf_counter() - record_start,
            "model_forward_pass_count": len(options),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "evaluation_metrics": {
                "mc1_correct": predicted == item["correct_option_index"],
                "mc2_true_probability_mass": sum(
                    probabilities[index]
                    for index in item["correct_option_indices"]
                ),
                "mc2_false_probability_mass": sum(
                    probabilities[index]
                    for index in range(len(probabilities))
                    if index not in item["correct_option_indices"]
                ),
                "option_count": len(options),
            },
        }
        prediction["record_checksum"] = checksum_record(
            prediction, "record_checksum"
        )
        probability = {"summary": _probability_summary(scores)}
        compact = {
            "version": VERSION,
            "experiment_id": experiment_id,
            "example_id": identifier,
            "dataset": "truthfulqa_mc",
            "domain": "all",
            **compact_features(selected_signature, probability),
        }
        compact["signature_checksum"] = checksum_record(
            compact, "signature_checksum"
        )
        ablation = {
            "version": VERSION,
            "experiment_id": experiment_id,
            "example_id": identifier,
            "dataset": "truthfulqa_mc",
            "domain": "all",
            "selected_option": secondary_features(selected_signature),
            "option_signatures": [
                {
                    "option_index": index,
                    "correct": options[index]["correct"],
                    "mc1_correct": options[index]["mc1_correct"],
                    "probability": probabilities[index],
                    "total_log_probability": scores[index][
                        "total_log_probability"
                    ],
                    "signature": secondary_features(scores[index]["signature"]),
                }
                for index in range(len(options))
            ],
        }
        ablation["ablation_checksum"] = checksum_record(
            ablation, "ablation_checksum"
        )
        filename = identifier.replace(":", "__") + ".json"
        prediction_path = prediction_directory / filename
        compact_path = compact_directory / filename
        ablation_path = ablation_directory / filename
        if not prediction_path.exists():
            atomic_json(prediction_path, prediction)
        if not compact_path.exists():
            atomic_json(compact_path, compact)
        if not ablation_path.exists():
            atomic_json(ablation_path, ablation)
        valid_predictions[identifier] = prediction
        valid_compact[identifier] = compact
        valid_ablation[identifier] = ablation
    identifiers = [item["example_id"] for item in manifest]
    checksums = {
        "predictions": compile_jsonl(
            prediction_directory,
            destination / "predictions/collection.jsonl",
            ordered_ids=identifiers,
            checksum_field="record_checksum",
        ),
        "compact_signatures": compile_jsonl(
            compact_directory,
            destination / "compact_signatures/collection.jsonl",
            ordered_ids=identifiers,
            checksum_field="signature_checksum",
        ),
        "signature_ablations": compile_jsonl(
            ablation_directory,
            destination / "signature_ablations/collection.jsonl",
            ordered_ids=identifiers,
            checksum_field="ablation_checksum",
        ),
    }
    atomic_json(
        destination / "verification_reports/artifact_checksums.json",
        {"version": VERSION, "experiment_id": experiment_id, "checksums": checksums},
    )
    return {"sample_count": len(manifest), "checksums": checksums}


def verify(manifest_path: Path, destination: Path) -> dict[str, Any]:
    expected = {item["example_id"] for item in read_jsonl(manifest_path)}
    result = {"expected": len(expected), "artifacts": {}}
    for directory, checksum_field in (
        ("predictions", "record_checksum"),
        ("compact_signatures", "signature_checksum"),
        ("signature_ablations", "ablation_checksum"),
    ):
        path = destination / directory / "collection.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        identifiers = {item["example_id"] for item in rows}
        result["artifacts"][directory] = {
            "valid_count": sum(
                validate_record_checksum(item, checksum_field) for item in rows
            ),
            "missing_count": len(expected - identifiers),
            "unexpected_count": len(identifiers - expected),
        }
    result["complete"] = all(
        item["valid_count"] == len(expected)
        and item["missing_count"] == 0
        and item["unexpected_count"] == 0
        for item in result["artifacts"].values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "verify"))
    parser.add_argument("--model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalized", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "collect":
        if args.model is None or args.normalized is None:
            parser.error("collect requires --model and --normalized")
        result = collect(
            args.model, args.manifest, args.normalized, args.destination
        )
    else:
        result = verify(args.manifest, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
