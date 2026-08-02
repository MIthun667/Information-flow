from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from usig.evaluation.audit_rules import (
    concise_alias_match,
    conservative_abstention,
    containment_diagnostics,
    evaluate_interpretation_segments,
)
from usig.experiment.collection import (
    MODEL_ID as FROZEN_MODEL_ID,
)
from usig.experiment.collection import (
    PROMPT_FAMILY,
    evaluate_response,
    load_prompt_specifications,
    sha256_file,
)
from usig.experiment.generation import (
    greedy_generate,
    load_generation_config,
    render_prompt,
)
from usig.experiment.hidden_states import (
    align_answer_hidden_states,
    completed_sequence_hidden_states,
    transition_matrices,
)
from usig.experiment.probabilities import extract_probability_features
from usig.experiment.records import (
    atomic_json,
    canonical_json,
    checksum_record,
    compile_jsonl,
    read_valid_records,
    validate_record_checksum,
)
from usig.experiment.signatures import calculate_signature

TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COMPACT_PROBABILITY_NAMES = (
    "mean_token_entropy",
    "maximum_token_entropy",
    "negative_mean_log_probability",
    "minimum_selected_token_log_probability",
    "selected_token_log_probability_std",
)
COMPACT_IFI_NAMES = (
    "scalar_ifi",
    "mean_token_instability",
    "maximum_token_instability",
    "token_instability_slope",
    "token_instability_roughness",
    "early_layer_transition_mean",
    "middle_layer_transition_mean",
    "late_layer_transition_mean",
    "normalized_peak_transition_layer",
    "layer_profile_roughness",
)


def compact_features(
    signature: dict[str, Any], probability: dict[str, Any]
) -> dict[str, Any]:
    tokens = signature["cosine_token_dynamics"]
    layers = signature["cosine_structured"]
    compact_ifi = {
        "scalar_ifi": signature["scalar_ifi"],
        "mean_token_instability": tokens["mean_token_instability"],
        "maximum_token_instability": tokens["maximum_token_instability"],
        "token_instability_slope": tokens["token_instability_slope"],
        "token_instability_roughness": tokens["token_instability_roughness"],
        "early_layer_transition_mean": layers["cosine_early_mean"],
        "middle_layer_transition_mean": layers["cosine_middle_mean"],
        "late_layer_transition_mean": layers["cosine_late_mean"],
        "normalized_peak_transition_layer": layers[
            "cosine_profile_normalized_maximum_position"
        ],
        "layer_profile_roughness": layers["cosine_profile_roughness"],
    }
    if tuple(compact_ifi) != COMPACT_IFI_NAMES:
        raise ValueError("Compact IFI feature membership changed")
    return {
        "feature_status": signature["feature_status"],
        "probability": {
            name: probability["summary"][name]
            for name in COMPACT_PROBABILITY_NAMES
        },
        "compact_ifi": compact_ifi,
        "definition_note": (
            "scalar_ifi equals token-instability population standard deviation; "
            "the duplicate standard-deviation feature is excluded"
        ),
    }


def secondary_features(signature: dict[str, Any]) -> dict[str, Any]:
    cosine = signature["cosine_structured"]
    return {
        "layer_profile_32": cosine["cosine_fixed_depth_profile"],
        "relative_transitions": signature["relative_structured"],
        "individual_token_dynamics": signature["cosine_token_dynamics"],
        "individual_layer_regions": {
            name: value
            for name, value in cosine.items()
            if "fixed_depth_profile" not in name
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_manifest(
    project_root: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = _read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("Manifest is empty")
    if len({item["example_id"] for item in manifest}) != len(manifest):
        raise ValueError("Duplicate example identifiers")
    if any(
        {"question", "context", "reference_answers"} & set(item)
        for item in manifest
    ):
        raise ValueError("Manifest contains canonical benchmark content")
    source_paths = list((project_root / "data/normalized").glob("*/*.jsonl"))
    by_checksum = {sha256_file(path): path for path in source_paths}
    needed = {item["source_checksum"] for item in manifest}
    if not needed <= set(by_checksum):
        raise ValueError("Manifest references an unknown canonical source")
    records = {}
    for checksum in needed:
        for record in _read_jsonl(by_checksum[checksum]):
            records[record["example_id"]] = record
    for item in manifest:
        record = records.get(item["example_id"])
        if record is None:
            raise ValueError(f"Missing canonical record: {item['example_id']}")
        digest = hashlib.sha256(
            (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        ).hexdigest()
        if digest != item["canonical_record_checksum"]:
            raise ValueError(f"Canonical checksum mismatch: {item['example_id']}")
    return manifest, records


def _cached_model_path(model_identifier: str) -> Path:
    cache_name = "models--" + model_identifier.replace("/", "--")
    snapshots = sorted(
        path
        for path in (
            Path.home() / ".cache/huggingface/hub" / cache_name / "snapshots"
        ).glob("*")
        if path.is_dir()
    )
    if len(snapshots) != 1:
        raise FileNotFoundError(
            f"Expected exactly one cached revision for {model_identifier}; "
            f"found {len(snapshots)}"
        )
    return snapshots[0]


def _load_model(model_identifier: str) -> tuple[Any, Any, dict[str, Any]]:
    model_path = _cached_model_path(model_identifier)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    revision = getattr(model.config, "_commit_hash", None) or model_path.name
    metadata = {
        "model_identifier": model_identifier,
        "model_revision": revision,
        "tokenizer_identifier": getattr(tokenizer, "name_or_path", str(model_path)),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or revision,
        "device": str(device),
        "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hidden_state_layer_count": int(model.config.num_hidden_layers),
        "hidden_dimension": int(model.config.hidden_size),
        "transformers_version": transformers.__version__,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
    }
    return model, tokenizer, metadata


def _identity(
    metadata: dict[str, Any],
    manifest_checksum: str,
    prompt_checksum: str,
    generation_config: dict[str, Any],
) -> str:
    configuration = {
        "model": metadata,
        "manifest_checksum": manifest_checksum,
        "prompt_checksum": prompt_checksum,
        "prompt_version": generation_config["prompt_version"],
        "prompt_versions": generation_config["prompt_versions"],
        "max_new_tokens": generation_config["max_new_tokens"],
        "greedy": True,
        "batch_size": 1,
        "completed_sequence_hidden_states": True,
        "compact_probability": COMPACT_PROBABILITY_NAMES,
        "compact_ifi": COMPACT_IFI_NAMES,
        "secondary_ablation_families": (
            "layer_profile_32",
            "relative_transitions",
            "individual_token_dynamics",
            "individual_layer_regions",
        ),
    }
    return "qwen_compact_" + hashlib.sha256(
        canonical_json(configuration).encode()
    ).hexdigest()[:16]


def _evaluation_diagnostics(record: dict[str, Any], response: str) -> dict[str, Any]:
    if record["dataset"] == "triviaqa":
        return {
            "concise_suffix": concise_alias_match(
                response, record["reference_answers"]
            ),
            "containment": containment_diagnostics(
                response, record["reference_answers"]
            ),
        }
    if record["dataset"] == "ambignq":
        interpretations = record["interpretations"] or [
            {
                "interpretation_id": "single_answer",
                "reference_answers": record["reference_answers"],
            }
        ]
        return evaluate_interpretation_segments(response, interpretations)
    if record["dataset"] == "truthfulqa":
        correct = containment_diagnostics(response, record["reference_answers"])
        incorrect = containment_diagnostics(
            response, record["incorrect_reference_answers"]
        )
        normalized = response.lower()
        refusal = any(
            phrase in normalized
            for phrase in ("i don't know", "i do not know", "cannot answer", "uncertain")
        )
        difference = correct["maximum_token_f1"] - incorrect["maximum_token_f1"]
        leaning = (
            "refusal_or_uncertainty"
            if refusal
            else "neither"
            if abs(difference) < 0.05
            else "correct_reference_leaning"
            if difference > 0
            else "incorrect_reference_leaning"
        )
        return {
            "lexical_category": leaning,
            "correct_references": correct,
            "incorrect_references": incorrect,
            "definitive_truthfulness_label": None,
        }
    return {}


def collect(
    project_root: Path,
    *,
    model_identifier: str,
    manifest_path: Path,
    output_destination: Path,
    stop_on_final_answer_line: bool = False,
    final_answer_window_tokens: int | None = None,
    stop_after_first_line: bool = False,
) -> dict[str, Any]:
    if model_identifier == FROZEN_MODEL_ID:
        raise ValueError("Refusing to write a new collection with the frozen pilot model")
    manifest, records = validate_manifest(project_root, manifest_path)
    prompts = load_prompt_specifications(project_root)
    generation_config = load_generation_config(project_root)
    model, tokenizer, model_metadata = _load_model(model_identifier)
    manifest_checksum = sha256_file(manifest_path)
    experiment_id = _identity(
        model_metadata, manifest_checksum, prompts["checksum"], generation_config
    )
    destination = output_destination.resolve()
    prediction_records = destination / "predictions/records"
    compact_records = destination / "compact_signatures/records"
    secondary_records = destination / "signature_ablations/records"
    valid_predictions = read_valid_records(prediction_records, "record_checksum")
    valid_compact = read_valid_records(compact_records, "signature_checksum")
    valid_secondary = read_valid_records(secondary_records, "ablation_checksum")
    metadata_path = destination / "extraction_metadata/experiment.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing["experiment_id"] != experiment_id:
            raise FileExistsError("Output destination belongs to another experiment")
    metadata_value = {
            "experiment_id": experiment_id,
            "creation_timestamp": datetime.now(UTC).isoformat(),
            "manifest": str(manifest_path),
            "manifest_checksum": manifest_checksum,
            "prompt_checksum": prompts["checksum"],
            "prompt_version": generation_config["prompt_version"],
            "prompt_versions": generation_config["prompt_versions"],
            "generation_config": str(generation_config["path"]),
            "max_new_tokens": generation_config["max_new_tokens"],
            "model": model_metadata,
            "compact_probability_features": COMPACT_PROBABILITY_NAMES,
            "compact_ifi_features": COMPACT_IFI_NAMES,
            "one_token_policy": "undefined_without_imputation",
            "stop_on_final_answer_line": stop_on_final_answer_line,
            "final_answer_window_tokens": final_answer_window_tokens,
            "stop_after_first_line": stop_after_first_line,
            "evaluator_version": "audit_rules_v3",
        }
    if not metadata_path.exists():
        atomic_json(metadata_path, metadata_value)
    processed = 0
    for item in manifest:
        example_id = item["example_id"]
        if (
            example_id in valid_predictions
            and example_id in valid_compact
            and example_id in valid_secondary
        ):
            continue
        record = records[example_id]
        family = PROMPT_FAMILY[record["dataset"]]
        # Dataset-specific templates allow repaired protocols without changing
        # the shared arithmetic/knowledge prompt families used by preserved runs.
        template = prompts["templates"].get(record["dataset"], prompts["templates"][family])
        expected_prompt_version = generation_config["prompt_versions"][
            record["dataset"]
        ]
        if template["template_id"] != expected_prompt_version:
            raise ValueError(
                f"Configured prompt version mismatch for {record['dataset']}: "
                f"{expected_prompt_version} != {template['template_id']}"
            )
        prompt = render_prompt(tokenizer, template, record)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        record_start = time.perf_counter()
        generation = greedy_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=generation_config["max_new_tokens"][record["dataset"]],
            stop_on_final_answer_line=stop_on_final_answer_line,
            stop_after_first_line=stop_after_first_line,
        )
        probability = extract_probability_features(
            generation["scores"], generation["generated_token_ids"]
        )
        extraction_start = time.perf_counter()
        hidden = completed_sequence_hidden_states(model, generation["full_token_ids"])
        aligned = align_answer_hidden_states(
            hidden,
            prompt_length=prompt["prompt_token_count"],
            generated_token_count=generation["generated_token_count"],
        )
        transitions = transition_matrices(aligned)
        signature = calculate_signature(transitions["cosine"], transitions["relative"])
        extraction_seconds = time.perf_counter() - extraction_start
        evaluation = evaluate_response(record, generation["response"])
        if record["dataset"] == "squad" and not record["answerable"]:
            abstention = conservative_abstention(generation["response"])
            evaluation["metrics"].update(
                {
                    "conservative_abstention": abstention,
                    "unsupported_answer": not abstention["abstained"],
                    "malformed_abstention": (
                        "unknown" in generation["response"].lower()
                        or "unanswer" in generation["response"].lower()
                    )
                    and not abstention["abstained"],
                }
            )
            evaluation["binary_correctness"] = abstention["abstained"]
            evaluation["binary_error"] = int(not abstention["abstained"])
        if record["dataset"] == "truthfulqa":
            evaluation["binary_correctness"] = None
            evaluation["binary_error"] = None
            evaluation["unresolved_label"] = True
        prediction = {
            "experiment_id": experiment_id,
            "example_id": example_id,
            "group_id": record["group_id"],
            "dataset": record["dataset"],
            "domain": record["split"],
            "model_identifier": model_identifier,
            "model_revision": model_metadata["model_revision"],
            "rendered_prompt": prompt["rendered_prompt"],
            "prompt_checksum": prompt["rendered_prompt_checksum"],
            "prompt_token_count": prompt["prompt_token_count"],
            "generated_token_count": generation["generated_token_count"],
            "generated_token_ids": generation["generated_token_ids"],
            "response": generation["response"],
            "normalized_response": evaluation["normalized_response"],
            "response_character_count": len(generation["response"]),
            "token_limit_reached": generation["token_limit_reached"],
            "stop_reason": generation["stop_reason"],
            "final_answer_stop_enabled": generation[
                "final_answer_stop_enabled"
            ],
            "final_answer_stop_detected": generation[
                "final_answer_stop_detected"
            ],
            "latency_seconds": generation["latency_seconds"],
            "hidden_state_extraction_seconds": extraction_seconds,
            "total_runtime_seconds": time.perf_counter() - record_start,
            "model_forward_pass_count": 2,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "evaluation_metrics": evaluation["metrics"],
            "evaluation_diagnostics": _evaluation_diagnostics(
                record, generation["response"]
            ),
            "binary_correctness": evaluation["binary_correctness"],
            "binary_error": evaluation["binary_error"],
            "unresolved_label": evaluation["unresolved_label"],
        }
        prediction["record_checksum"] = checksum_record(
            prediction, "record_checksum"
        )
        compact = {
            "experiment_id": experiment_id,
            "example_id": example_id,
            "dataset": record["dataset"],
            "domain": record["split"],
            **compact_features(signature, probability),
        }
        compact["signature_checksum"] = checksum_record(
            compact, "signature_checksum"
        )
        secondary = {
            "experiment_id": experiment_id,
            "example_id": example_id,
            "dataset": record["dataset"],
            "domain": record["split"],
            **secondary_features(signature),
        }
        if final_answer_window_tokens is not None:
            window = aligned[-final_answer_window_tokens:]
            window_transitions = transition_matrices(window)
            window_signature = calculate_signature(
                window_transitions["cosine"], window_transitions["relative"]
            )
            secondary["final_answer_window"] = {
                "window_token_count": int(window.shape[0]),
                **secondary_features(window_signature),
                "compact_ifi": compact_features(
                    window_signature, probability
                )["compact_ifi"],
            }
        secondary["ablation_checksum"] = checksum_record(
            secondary, "ablation_checksum"
        )
        del hidden, aligned, transitions
        if "window_transitions" in locals():
            del window_transitions
        filename = f"{example_id.replace(':', '__')}.json"
        atomic_json(prediction_records / filename, prediction)
        atomic_json(compact_records / filename, compact)
        atomic_json(secondary_records / filename, secondary)
        valid_predictions[example_id] = prediction
        valid_compact[example_id] = compact
        valid_secondary[example_id] = secondary
        processed += 1
        atomic_json(
            destination / "extraction_metadata/completion.json",
            {
                "experiment_id": experiment_id,
                "expected": len(manifest),
                "valid_predictions": len(valid_predictions),
                "valid_compact_signatures": len(valid_compact),
                "valid_secondary_signatures": len(valid_secondary),
                "last_example_id": example_id,
            },
        )
    identifiers = [item["example_id"] for item in manifest]
    checksums = {}
    if (
        len(valid_predictions)
        == len(valid_compact)
        == len(valid_secondary)
        == len(manifest)
    ):
        checksums = {
            "predictions": compile_jsonl(
                prediction_records,
                destination / "predictions/collection.jsonl",
                ordered_ids=identifiers,
                checksum_field="record_checksum",
            ),
            "compact_signatures": compile_jsonl(
                compact_records,
                destination / "compact_signatures/collection.jsonl",
                ordered_ids=identifiers,
                checksum_field="signature_checksum",
            ),
            "signature_ablations": compile_jsonl(
                secondary_records,
                destination / "signature_ablations/collection.jsonl",
                ordered_ids=identifiers,
                checksum_field="ablation_checksum",
            ),
        }
        atomic_json(
            destination / "verification_reports/artifact_checksums.json",
            {"experiment_id": experiment_id, "checksums": checksums},
        )
    return {
        "experiment_id": experiment_id,
        "processed": processed,
        "expected": len(manifest),
        "valid_predictions": len(valid_predictions),
        "valid_compact_signatures": len(valid_compact),
        "checksums": checksums,
    }


def verify(
    project_root: Path, manifest_path: Path, output_destination: Path
) -> dict[str, Any]:
    manifest, _ = validate_manifest(project_root, manifest_path)
    identifiers = {item["example_id"] for item in manifest}
    specifications = (
        ("predictions", "record_checksum"),
        ("compact_signatures", "signature_checksum"),
        ("signature_ablations", "ablation_checksum"),
    )
    result = {"expected": len(manifest), "artifacts": {}}
    for directory, checksum_field in specifications:
        record_directory = output_destination / directory / "records"
        paths = sorted(record_directory.glob("*.json"))
        decoded = []
        corrupt_paths = []
        non_finite_paths = []
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                corrupt_paths.append(str(path))
                continue
            if not validate_record_checksum(record, checksum_field):
                corrupt_paths.append(str(path))
                continue
            decoded.append(record)
            values: list[Any] = [record]
            while values:
                value = values.pop()
                if isinstance(value, dict):
                    values.extend(value.values())
                elif isinstance(value, list):
                    values.extend(value)
                elif isinstance(value, float) and not torch.isfinite(
                    torch.tensor(value)
                ):
                    non_finite_paths.append(str(path))
                    break
        records = read_valid_records(
            record_directory, checksum_field
        )
        decoded_ids = [item["example_id"] for item in decoded]
        result["artifacts"][directory] = {
            "valid_count": len(records),
            "missing_count": len(identifiers - set(records)),
            "unexpected_count": len(set(records) - identifiers),
            "duplicate_identifier_count": len(decoded_ids) - len(set(decoded_ids)),
            "checksum_failure_count": len(corrupt_paths),
            "non_finite_feature_count": len(set(non_finite_paths)),
            "corrupt_paths": corrupt_paths,
        }
    result["complete"] = all(
        item["valid_count"] == len(manifest)
        and item["missing_count"] == 0
        and item["unexpected_count"] == 0
        and item["duplicate_identifier_count"] == 0
        and item["checksum_failure_count"] == 0
        and item["non_finite_feature_count"] == 0
        for item in result["artifacts"].values()
    )
    return result


def partial_artifacts(
    output_destination: Path, *, clean_confirmed: bool
) -> dict[str, Any]:
    recognized = []
    checksum_fields = {
        "predictions": "record_checksum",
        "compact_signatures": "signature_checksum",
        "signature_ablations": "ablation_checksum",
    }
    for directory, checksum_field in checksum_fields.items():
        record_directory = output_destination / directory / "records"
        for path in sorted(record_directory.glob("*")):
            if path.name.startswith("tmp"):
                recognized.append({"path": str(path), "reason": "temporary_file"})
                continue
            if path.suffix != ".json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                recognized.append({"path": str(path), "reason": "invalid_json"})
                continue
            if not validate_record_checksum(record, checksum_field):
                recognized.append(
                    {"path": str(path), "reason": "checksum_mismatch"}
                )
    removed = []
    if clean_confirmed:
        for item in recognized:
            path = Path(item["path"])
            path.unlink()
            removed.append(str(path))
    return {
        "output_destination": str(output_destination),
        "recognized_partial_count": len(recognized),
        "recognized_partials": recognized,
        "removed": removed,
        "protected_artifact_classes": [
            "compiled predictions",
            "compiled signatures",
            "metadata",
            "metrics",
            "verification reports",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect compact Qwen signatures.")
    parser.add_argument(
        "action", choices=("collect", "resume", "verify", "partials")
    )
    parser.add_argument("--model")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-destination", type=Path, required=True)
    parser.add_argument("--clean-confirmed", action="store_true")
    parser.add_argument("--stop-on-final-answer-line", action="store_true")
    parser.add_argument("--final-answer-window-tokens", type=int)
    parser.add_argument("--stop-after-first-line", action="store_true")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    if args.action == "partials":
        result = partial_artifacts(
            args.output_destination, clean_confirmed=args.clean_confirmed
        )
    elif args.action == "verify":
        if args.manifest is None:
            parser.error("--manifest is required for verification")
        result = verify(
            args.project_root, args.manifest, args.output_destination
        )
    else:
        if args.model is None or args.manifest is None:
            parser.error("--model and --manifest are required for collection")
        result = collect(
            args.project_root,
            model_identifier=args.model,
            manifest_path=args.manifest,
            output_destination=args.output_destination,
            stop_on_final_answer_line=args.stop_on_final_answer_line,
            final_answer_window_tokens=args.final_answer_window_tokens,
            stop_after_first_line=args.stop_after_first_line,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.action == "verify" and not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
