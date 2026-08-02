"""Discover model-conditioned reasoning-uncertainty candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from usig.experiment.prepare_reasoning_curated_pilot import (
    difficulty_score,
    transform_record,
    triviality_reasons,
    validate_source_record,
)
from usig.experiment.screen_reasoning_interventions import (
    build_chat_prompt,
    generate_one,
    load_yaml,
    resolve_dtype,
    set_deterministic_seed,
    sha256_text,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/uncertainty_flow/reasoning_discovery.yaml"
)


def load_source_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load and validate normalized IFI-ARITH records."""

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            payload = json.loads(line)

            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )

            validate_source_record(payload)
            records.append(payload)

    return records


def parse_sampling_quotas(
    payload: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, int]]:
    quotas: list[tuple[str, str, int]] = []

    for item in payload:
        domain = str(item["domain"])
        operation = str(item["operation"])
        count = int(item["count"])

        if count <= 0:
            raise ValueError(
                f"sampling quota must be positive: "
                f"{domain}:{operation}={count}"
            )

        quotas.append((domain, operation, count))

    if not quotas:
        raise ValueError("at least one sampling quota is required")

    if len(quotas) != len(
        {(domain, operation) for domain, operation, _ in quotas}
    ):
        raise ValueError("sampling quotas contain duplicate buckets")

    return quotas


def bucket_key(record: Mapping[str, Any]) -> str:
    return (
        f"{record['domain']}:"
        f"{record['metadata']['operation']}"
    )


def select_discovery_pool(
    records_by_domain: Mapping[str, Iterable[dict[str, Any]]],
    quotas: Sequence[tuple[str, str, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, quota-controlled candidate pool."""

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    eligible = Counter()

    for expected_domain, records in records_by_domain.items():
        for record in records:
            actual_domain = str(record["domain"])

            if actual_domain != expected_domain:
                raise ValueError(
                    f"record domain {actual_domain!r} does not match "
                    f"source domain {expected_domain!r}"
                )

            operation = str(record["metadata"]["operation"])
            reasons = triviality_reasons(record)

            if reasons:
                rejected.update(reasons)
                continue

            buckets[(actual_domain, operation)].append(record)
            eligible[f"{actual_domain}:{operation}"] += 1

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_answers: set[str] = set()
    duplicate_answer_skips = Counter()

    for domain, operation, required_count in quotas:
        candidates = sorted(
            buckets[(domain, operation)],
            key=difficulty_score,
        )

        bucket_selected: list[dict[str, Any]] = []

        for candidate in candidates:
            example_id = str(candidate["example_id"])
            answer = str(candidate["metadata"]["expected_answer"])

            if example_id in selected_ids:
                continue

            if answer in selected_answers:
                duplicate_answer_skips[
                    f"{domain}:{operation}"
                ] += 1
                continue

            bucket_selected.append(candidate)
            selected_ids.add(example_id)
            selected_answers.add(answer)

            if len(bucket_selected) == required_count:
                break

        if len(bucket_selected) < required_count:
            raise ValueError(
                f"{domain}:{operation} produced only "
                f"{len(bucket_selected)} unique candidates; "
                f"{required_count} are required"
            )

        selected.extend(bucket_selected)

    selected.sort(
        key=lambda record: (
            str(record["domain"]),
            str(record["metadata"]["operation"]),
            str(record["example_id"]),
        )
    )

    audit = {
        "eligible_counts": dict(sorted(eligible.items())),
        "rejection_counts": dict(sorted(rejected.items())),
        "duplicate_answer_skips": dict(
            sorted(duplicate_answer_skips.items())
        ),
        "selected_count": len(selected),
        "selected_bucket_counts": dict(
            sorted(Counter(bucket_key(record) for record in selected).items())
        ),
    }

    return selected, audit


@torch.inference_mode()
def score_gold_answer(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    gold_answer: str,
) -> dict[str, float | int | None]:
    """Score only the numeric gold-answer tokens after the FINAL prefix."""

    formatted_prompt = build_chat_prompt(tokenizer, prompt)
    scoring_prefix = formatted_prompt + "FINAL: "
    target_text = str(gold_answer)

    prefix_tokens = tokenizer(
        scoring_prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    full_tokens = tokenizer(
        scoring_prefix + target_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    prefix_length = int(prefix_tokens.shape[1])
    target_tokens = full_tokens[:, prefix_length:]

    if target_tokens.shape[1] == 0:
        raise ValueError("gold answer produced zero target tokens")

    model_device = next(model.parameters()).device
    full_tokens = full_tokens.to(model_device)
    target_tokens = target_tokens.to(model_device)

    attention_mask = torch.ones_like(full_tokens)

    output = model(
        input_ids=full_tokens,
        attention_mask=attention_mask,
        use_cache=False,
    )

    target_length = int(target_tokens.shape[1])

    # The logits immediately before each target token predict that token.
    target_logits = output.logits[
        :,
        prefix_length - 1 : prefix_length + target_length - 1,
        :,
    ].float()

    if target_logits.shape[1] != target_length:
        raise RuntimeError(
            "gold-answer scoring produced a token-length mismatch"
        )

    log_probabilities = torch.log_softmax(
        target_logits,
        dim=-1,
    )

    selected_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1),
    ).squeeze(-1)

    token_probabilities = selected_log_probabilities.exp()

    return {
        "gold_token_count": target_length,
        "gold_sequence_log_probability": float(
            selected_log_probabilities.sum().item()
        ),
        "gold_mean_token_log_probability": float(
            selected_log_probabilities.mean().item()
        ),
        "gold_mean_token_probability": float(
            token_probabilities.mean().item()
        ),
        "gold_minimum_token_probability": float(
            token_probabilities.min().item()
        ),
        "gold_scoring_prefix": "FINAL: ",
    }

def classify_original_record(
    record: Mapping[str, Any],
    *,
    low_gold_probability_threshold: float,
) -> str:
    """Classify one original-only screening result."""

    if not bool(record["is_correct"]):
        return "wrong"

    probability = record.get("gold_mean_token_probability")

    if not isinstance(probability, (int, float)):
        return "missing_gold_score"

    if not math.isfinite(float(probability)):
        return "missing_gold_score"

    if float(probability) < low_gold_probability_threshold:
        return "low_gold_likelihood"

    return "easy"


def uncertainty_sort_key(
    record: Mapping[str, Any],
) -> tuple[int, float, float, str]:
    """Rank wrong and low-gold-likelihood examples first."""

    status_priority = {
        "wrong": 0,
        "low_gold_likelihood": 1,
        "missing_gold_score": 2,
        "easy": 3,
    }

    status = str(record["discovery_status"])

    gold_log_probability = record.get(
        "gold_mean_token_log_probability"
    )
    if not isinstance(gold_log_probability, (int, float)):
        gold_log_probability = 0.0

    generated_entropy = record.get("mean_token_entropy")
    if not isinstance(generated_entropy, (int, float)):
        generated_entropy = 0.0

    return (
        status_priority.get(status, 99),
        float(gold_log_probability),
        -float(generated_entropy),
        str(record["example_id"]),
    )


def select_shortlist(
    records: Sequence[dict[str, Any]],
    *,
    shortlist_size: int,
    minimum_per_bucket: int,
    maximum_per_bucket: int,
) -> list[dict[str, Any]]:
    """Select a diverse shortlist from model-conditioned results."""

    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")

    if minimum_per_bucket < 0:
        raise ValueError("minimum_per_bucket cannot be negative")

    if maximum_per_bucket < minimum_per_bucket:
        raise ValueError(
            "maximum_per_bucket must be at least minimum_per_bucket"
        )

    useful = [
        record
        for record in records
        if record["discovery_status"]
        in {
            "wrong",
            "low_gold_likelihood",
            "missing_gold_score",
        }
    ]

    useful.sort(key=uncertainty_sort_key)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in useful:
        by_bucket[str(record["bucket"])].append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    bucket_counts = Counter()

    # First preserve operation/domain coverage.
    for bucket in sorted(by_bucket):
        for record in by_bucket[bucket][:minimum_per_bucket]:
            if len(selected) >= shortlist_size:
                break

            example_id = str(record["example_id"])

            if example_id in selected_ids:
                continue

            selected.append(record)
            selected_ids.add(example_id)
            bucket_counts[bucket] += 1

    # Then fill globally by uncertainty rank.
    for record in useful:
        if len(selected) >= shortlist_size:
            break

        example_id = str(record["example_id"])
        bucket = str(record["bucket"])

        if example_id in selected_ids:
            continue

        if bucket_counts[bucket] >= maximum_per_bucket:
            continue

        selected.append(record)
        selected_ids.add(example_id)
        bucket_counts[bucket] += 1

    selected.sort(key=uncertainty_sort_key)

    return selected


def run_discovery(
    config_path: Path,
    *,
    overwrite: bool,
) -> None:
    config = load_yaml(config_path)

    output_records = (
        PROJECT_ROOT / config["output"]["records"]
    )
    output_shortlist = (
        PROJECT_ROOT / config["output"]["shortlist"]
    )
    output_summary = (
        PROJECT_ROOT / config["output"]["summary"]
    )

    for path in (
        output_records,
        output_shortlist,
        output_summary,
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}"
            )

    seed = int(config["experiment"]["seed"])
    set_deterministic_seed(seed)

    records_by_domain = {
        domain: load_source_jsonl(
            PROJECT_ROOT / relative_path
        )
        for domain, relative_path in config["input"].items()
    }

    quotas = parse_sampling_quotas(
        config["sampling"]["quotas"]
    )

    candidates, sampling_audit = select_discovery_pool(
        records_by_domain,
        quotas,
    )

    model_config = config["model"]
    generation_config = config["generation"]
    selection_config = config["selection"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        trust_remote_code=bool(
            model_config.get("trust_remote_code", False)
        ),
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        dtype=resolve_dtype(model_config["dtype"]),
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=bool(
            model_config.get("trust_remote_code", False)
        ),
    )
    model.eval()

    screening_records: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        group = transform_record(
            candidate,
            group_index=index,
        )

        prompt = str(group["original"]["prompt"])
        gold_answers = [
            str(answer)
            for answer in group["gold_answers"]
        ]
        primary_gold = gold_answers[0]

        generated = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gold_answers=gold_answers,
            generation_config=generation_config,
        )

        gold_metrics = score_gold_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gold_answer=primary_gold,
        )

        record = {
            "candidate_index": index,
            "example_id": str(candidate["example_id"]),
            "source_id": str(candidate.get("source_id", "")),
            "domain": str(candidate["domain"]),
            "operation": str(
                candidate["metadata"]["operation"]
            ),
            "bucket": bucket_key(candidate),
            "expression": str(
                candidate["metadata"]["expression"]
            ),
            "gold_answers": gold_answers,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "model_id": model_config["model_id"],
            "seed": seed,
            "difficulty_score": list(
                difficulty_score(candidate)[:-1]
            ),
            **asdict(generated),
            **gold_metrics,
        }

        record["discovery_status"] = classify_original_record(
            record,
            low_gold_probability_threshold=float(
                selection_config[
                    "low_gold_probability_threshold"
                ]
            ),
        )

        screening_records.append(record)

        print(
            f"[{index + 1:03d}/{len(candidates):03d}] "
            f"{record['bucket']} | "
            f"{record['expression']} | "
            f"prediction={record['normalized_prediction']} | "
            f"correct={record['is_correct']} | "
            f"gold_p={record['gold_mean_token_probability']:.6f} | "
            f"status={record['discovery_status']}"
        )

    shortlist = select_shortlist(
        screening_records,
        shortlist_size=int(
            selection_config["shortlist_size"]
        ),
        minimum_per_bucket=int(
            selection_config["minimum_per_bucket"]
        ),
        maximum_per_bucket=int(
            selection_config["maximum_per_bucket"]
        ),
    )

    write_jsonl(
        output_records,
        screening_records,
    )
    write_jsonl(
        output_shortlist,
        shortlist,
    )

    status_counts = Counter(
        record["discovery_status"]
        for record in screening_records
    )
    shortlist_bucket_counts = Counter(
        record["bucket"]
        for record in shortlist
    )
    shortlist_status_counts = Counter(
        record["discovery_status"]
        for record in shortlist
    )

    bucket_accuracy: dict[str, float] = {}

    records_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in screening_records:
        records_by_bucket[record["bucket"]].append(record)

    for bucket, bucket_records in sorted(records_by_bucket.items()):
        bucket_accuracy[bucket] = (
            sum(bool(record["is_correct"]) for record in bucket_records)
            / len(bucket_records)
        )

    summary = {
        "experiment": config["experiment"],
        "model": model_config,
        "generation": generation_config,
        "candidate_count": len(screening_records),
        "shortlist_count": len(shortlist),
        "overall_accuracy": (
            sum(
                bool(record["is_correct"])
                for record in screening_records
            )
            / len(screening_records)
        ),
        "bucket_accuracy": bucket_accuracy,
        "status_counts": dict(sorted(status_counts.items())),
        "shortlist_status_counts": dict(
            sorted(shortlist_status_counts.items())
        ),
        "shortlist_bucket_counts": dict(
            sorted(shortlist_bucket_counts.items())
        ),
        "sampling_audit": sampling_audit,
        "records_path": str(output_records),
        "shortlist_path": str(output_shortlist),
    }

    write_json(
        output_summary,
        summary,
    )

    print("\nDiscovery summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover model-conditioned arithmetic reasoning candidates."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    run_discovery(
        arguments.config,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
