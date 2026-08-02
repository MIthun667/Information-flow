from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from usig.evaluation import (
    evaluate_aliases,
    evaluate_ambignq,
    evaluate_arithmetic_answer,
    evaluate_squad,
    evaluate_truthfulqa,
)
from usig.experiment.generation import MAX_NEW_TOKENS, greedy_generate, render_prompt
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
)
from usig.experiment.signatures import calculate_signature

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EXPECTED_MANIFEST_CHECKSUM = "52c5ceeb1707a20f537deeb54e1d24d3f6484f96bd55d8f1ebd70339a3c518c4"
PROMPT_FAMILY = {
    "ifi_arith": "reasoning",
    "gsm8k": "reasoning",
    "truthfulqa": "knowledge",
    "triviaqa": "knowledge",
    "ambignq": "ambiguity",
    "squad": "context_grounded",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_collection(project_root: Path) -> dict[str, Any]:
    manifest_path = (
        project_root / "data/manifests/pilots/six_benchmark_seed2026_n600.jsonl"
    )
    checksum = sha256_file(manifest_path)
    if checksum != EXPECTED_MANIFEST_CHECKSUM:
        raise ValueError(f"Manifest checksum mismatch: {checksum}")
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != 600:
        raise ValueError("Combined manifest must contain 600 records")
    counts: dict[str, int] = {}
    for item in manifest:
        counts[item["dataset"]] = counts.get(item["dataset"], 0) + 1
        if {"question", "context", "reference_answers"} & item.keys():
            raise ValueError("Manifest embeds canonical benchmark content")
    if set(counts.values()) != {100} or len(counts) != 6:
        raise ValueError(f"Invalid per-benchmark manifest counts: {counts}")
    if len({item["example_id"] for item in manifest}) != 600:
        raise ValueError("Duplicate canonical IDs in combined manifest")
    if len({item["group_id"] for item in manifest}) != 600:
        raise ValueError("Duplicate group IDs in combined manifest")

    normalized_root = project_root / "data/normalized"
    source_paths = list(normalized_root.glob("*/*.jsonl"))
    source_by_checksum = {sha256_file(path): path for path in source_paths}
    records: dict[str, dict[str, Any]] = {}
    for item in manifest:
        source_path = source_by_checksum.get(item["source_checksum"])
        if source_path is None:
            raise ValueError(f"Unknown canonical source checksum: {item['source_checksum']}")
        if item["example_id"] not in records:
            for record in _read_jsonl(source_path):
                records.setdefault(record["example_id"], record)
        record = records.get(item["example_id"])
        if record is None:
            raise ValueError(f"Missing canonical record: {item['example_id']}")
        digest = hashlib.sha256(
            (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        ).hexdigest()
        if digest != item["canonical_record_checksum"]:
            raise ValueError(f"Canonical checksum mismatch: {item['example_id']}")
    return {
        "manifest_path": manifest_path,
        "manifest_checksum": checksum,
        "manifest": manifest,
        "records": records,
        "counts": counts,
    }


def load_prompt_specifications(project_root: Path) -> dict[str, Any]:
    path = project_root / "config/prompts/benchmark_prompts.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for template in config["templates"].values():
        actual = hashlib.sha256(template["text"].encode()).hexdigest()
        if actual != template["checksum"]:
            raise ValueError("Semantic prompt checksum mismatch")
    return {
        "path": path,
        "checksum": sha256_file(path),
        "templates": config["templates"],
    }


def load_model() -> tuple[Any, Any, dict[str, Any]]:
    snapshot_root = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    )
    snapshots = sorted(path for path in snapshot_root.glob("*") if path.is_dir())
    if len(snapshots) != 1:
        raise FileNotFoundError(
            "Expected exactly one locally cached Qwen2.5-0.5B-Instruct revision; "
            f"found {len(snapshots)}"
        )
    model_path = snapshots[0]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    revision = getattr(model.config, "_commit_hash", None) or model_path.name
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or revision
    gpu = None
    memory = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        free, total = torch.cuda.mem_get_info(device)
        memory = {"free": free, "total": total}
    metadata = {
        "model_identifier": MODEL_ID,
        "model_revision": revision,
        "tokenizer_identifier": MODEL_ID,
        "tokenizer_revision": tokenizer_revision,
        "model_configuration": model.config.to_dict(),
        "hidden_size": model.config.hidden_size,
        "transformer_layers": model.config.num_hidden_layers,
        "vocabulary_size": model.config.vocab_size,
        "chat_template_available": bool(tokenizer.chat_template),
        "padding_side": tokenizer.padding_side,
        "special_tokens": tokenizer.special_tokens_map,
        "dtype": str(dtype),
        "device": str(device),
        "transformers_version": transformers.__version__,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu,
        "gpu_memory": memory,
        "platform": platform.platform(),
    }
    return model, tokenizer, metadata


def experiment_identity(
    model_metadata: dict[str, Any],
    manifest_checksum: str,
    prompt_checksum: str,
) -> str:
    configuration = {
        "model_identifier": model_metadata["model_identifier"],
        "model_revision": model_metadata["model_revision"],
        "tokenizer_revision": model_metadata["tokenizer_revision"],
        "manifest_checksum": manifest_checksum,
        "prompt_checksum": prompt_checksum,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "features": {
            "interior_proportions": [0.1, 0.9],
            "scalar_ifi_population_std": True,
            "fixed_depth_profile_positions": 32,
            "full_hidden_tensor_storage": False,
        },
    }
    return "qwen_ifi_" + hashlib.sha256(canonical_json(configuration).encode()).hexdigest()[:16]


def evaluate_response(record: dict[str, Any], response: str) -> dict[str, Any]:
    dataset = record["dataset"]
    binary: bool | None
    unresolved = False
    if dataset in {"ifi_arith", "gsm8k"}:
        metrics = evaluate_arithmetic_answer(
            response,
            record["reference_answers"][0],
            final_answer=dataset == "gsm8k",
        )
        binary = bool(metrics["exact_match"])
        normalized = metrics["normalized_answer"]
    elif dataset == "triviaqa":
        metrics = evaluate_aliases(response, record["reference_answers"])
        binary = bool(metrics["exact_match"])
        normalized = metrics["normalized_prediction"]
    elif dataset == "ambignq":
        interpretations = record["interpretations"]
        if not interpretations:
            # AmbigNQ-light keeps single-answer aliases at record level.
            interpretations = [
                {
                    "interpretation_id": "single_answer",
                    "reference_answers": record["reference_answers"],
                }
            ]
        metrics = evaluate_ambignq(response, interpretations)
        binary = bool(metrics["any_interpretation_exact_match"])
        normalized = response.strip()
    elif dataset == "squad":
        metrics = evaluate_squad(
            response, record["reference_answers"], answerable=record["answerable"]
        )
        binary = bool(metrics["combined_correct"])
        normalized = response.strip()
    elif dataset == "truthfulqa":
        metrics = evaluate_truthfulqa(
            response,
            record["reference_answers"],
            record["incorrect_reference_answers"],
        )
        status = metrics["status"]
        binary = (
            True
            if status == "matched_correct_reference"
            else False
            if status == "matched_incorrect_reference"
            else None
        )
        unresolved = binary is None
        normalized = response.strip()
    else:
        raise ValueError(f"Unsupported evaluation dataset: {dataset}")
    # Keep canonical gold strings in canonical data only. Evaluation artifacts
    # retain derived scores/statuses but not reference answers or matched aliases.
    metrics.pop("reference_answer", None)
    metrics.pop("matched_aliases", None)
    return {
        "metrics": metrics,
        "binary_correctness": binary,
        "binary_error": None if binary is None else int(not binary),
        "unresolved_label": unresolved,
        "normalized_response": normalized,
    }


def collect(project_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    validated = validate_collection(project_root)
    prompts = load_prompt_specifications(project_root)
    model, tokenizer, model_metadata = load_model()
    experiment_id = experiment_identity(
        model_metadata, validated["manifest_checksum"], prompts["checksum"]
    )
    output_root = project_root / "outputs"
    prediction_records = output_root / "predictions" / experiment_id / "records"
    signature_records = output_root / "signatures" / experiment_id / "records"
    valid_predictions = read_valid_records(prediction_records, "record_checksum")
    valid_signatures = read_valid_records(signature_records, "signature_checksum")
    for example_id, prediction in valid_predictions.items():
        metrics = prediction["evaluation_metrics"]
        if "reference_answer" in metrics or "matched_aliases" in metrics:
            metrics.pop("reference_answer", None)
            metrics.pop("matched_aliases", None)
            prediction["record_checksum"] = checksum_record(
                prediction, "record_checksum"
            )
            atomic_json(
                prediction_records / f"{example_id.replace(':', '__')}.json",
                prediction,
            )
    metadata = {
        "experiment_id": experiment_id,
        "execution_timestamp": datetime.now(UTC).isoformat(),
        "manifest_checksum": validated["manifest_checksum"],
        "prompt_specification_checksum": prompts["checksum"],
        "model": model_metadata,
        "generation": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS},
        "features": {
            "hidden_state_method": "completed_sequence_forward_pass",
            "embedding_level_excluded": True,
            "interior_transition_proportions": [0.1, 0.9],
            "scalar_ifi_standard_deviation": "population",
            "one_token_policy": "null_with_insufficient_tokens",
        },
    }
    atomic_json(output_root / "metadata" / experiment_id / "experiment.json", metadata)

    processed = 0
    for item in validated["manifest"][:limit]:
        example_id = item["example_id"]
        if example_id in valid_predictions and example_id in valid_signatures:
            continue
        record = validated["records"][example_id]
        family = PROMPT_FAMILY[record["dataset"]]
        prompt = render_prompt(tokenizer, prompts["templates"][family], record)
        generation = greedy_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=MAX_NEW_TOKENS[record["dataset"]],
        )
        probability = extract_probability_features(
            generation["scores"], generation["generated_token_ids"]
        )
        hidden = completed_sequence_hidden_states(model, generation["full_token_ids"])
        aligned = align_answer_hidden_states(
            hidden,
            prompt_length=prompt["prompt_token_count"],
            generated_token_count=generation["generated_token_count"],
        )
        transitions = transition_matrices(aligned)
        signature = calculate_signature(transitions["cosine"], transitions["relative"])
        evaluation = evaluate_response(record, generation["response"])
        prediction = {
            "experiment_id": experiment_id,
            "example_id": example_id,
            "group_id": record["group_id"],
            "dataset": record["dataset"],
            "task_family": record["task_family"],
            "source_split": record["split"],
            "model_identifier": MODEL_ID,
            "model_revision": model_metadata["model_revision"],
            "prompt_template_id": prompts["templates"][family]["template_id"],
            "semantic_template_checksum": prompts["templates"][family]["checksum"],
            "rendered_prompt": prompt["rendered_prompt"],
            "prompt_checksum": prompt["rendered_prompt_checksum"],
            "chat_template_status": prompt["chat_template_status"],
            "prompt_token_count": prompt["prompt_token_count"],
            "generated_token_ids": generation["generated_token_ids"],
            "generated_token_count": generation["generated_token_count"],
            "raw_response": generation["raw_response"],
            "response": generation["response"],
            "normalized_response": evaluation["normalized_response"],
            "stop_reason": generation["stop_reason"],
            "token_limit_reached": generation["token_limit_reached"],
            "total_token_count": generation["total_token_count"],
            "latency_seconds": generation["latency_seconds"],
            "peak_allocated_gpu_memory": generation["peak_allocated_gpu_memory"],
            "evaluation_metrics": evaluation["metrics"],
            "binary_correctness": evaluation["binary_correctness"],
            "binary_error": evaluation["binary_error"],
            "unresolved_label": evaluation["unresolved_label"],
        }
        prediction["record_checksum"] = checksum_record(prediction, "record_checksum")
        probability["summary"]["prompt_token_count"] = prompt["prompt_token_count"]
        signature_record = {
            "experiment_id": experiment_id,
            "example_id": example_id,
            "dataset": record["dataset"],
            "model_identifier": MODEL_ID,
            "generated_token_count": generation["generated_token_count"],
            "feature_status": signature["feature_status"],
            "scalar_ifi": signature["scalar_ifi"],
            "signature": signature,
            "probability_summaries": probability["summary"],
            "token_probability_features": probability["tokens"],
            "transition_shape": list(transitions["cosine"].shape),
            "hidden_state_levels": len(hidden),
            "hidden_size": int(aligned.shape[-1]),
        }
        signature_record["signature_checksum"] = checksum_record(
            signature_record, "signature_checksum"
        )
        atomic_json(prediction_records / f"{example_id.replace(':', '__')}.json", prediction)
        atomic_json(signature_records / f"{example_id.replace(':', '__')}.json", signature_record)
        valid_predictions[example_id] = prediction
        valid_signatures[example_id] = signature_record
        processed += 1
        atomic_json(
            output_root / "metadata" / experiment_id / "completion.json",
            {
                "experiment_id": experiment_id,
                "valid_predictions": len(valid_predictions),
                "valid_signatures": len(valid_signatures),
                "last_example_id": example_id,
            },
        )
    ordered_ids = [item["example_id"] for item in validated["manifest"]]
    if len(valid_predictions) == 600 and len(valid_signatures) == 600:
        prediction_checksum = compile_jsonl(
            prediction_records,
            output_root / "predictions" / f"{experiment_id}.jsonl",
            ordered_ids=ordered_ids,
            checksum_field="record_checksum",
        )
        signature_checksum = compile_jsonl(
            signature_records,
            output_root / "signatures" / f"{experiment_id}.jsonl",
            ordered_ids=ordered_ids,
            checksum_field="signature_checksum",
        )
    else:
        prediction_checksum = signature_checksum = None
    return {
        "experiment_id": experiment_id,
        "processed_this_run": processed,
        "valid_predictions": len(valid_predictions),
        "valid_signatures": len(valid_signatures),
        "prediction_checksum": prediction_checksum,
        "signature_checksum": signature_checksum,
    }
