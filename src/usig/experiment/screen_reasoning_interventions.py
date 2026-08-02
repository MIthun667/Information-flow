"""Screen reasoning interventions using deterministic Qwen generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/uncertainty_flow/reasoning_screening.yaml"
)

_VARIANTS = (
    "original",
    "resolved",
    "irrelevant_control",
)

_INTEGER_PATTERN = re.compile(r"[-+]?\d[\d,]*")
_FINAL_PATTERN = re.compile(
    r"FINAL\s*:\s*([-+]?\d[\d,]*)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Behavioral and confidence measurements for one generated response."""

    generated_text: str
    normalized_prediction: str | None
    is_correct: bool
    generated_token_count: int
    reached_generation_limit: bool

    sequence_log_probability: float | None
    mean_token_log_probability: float | None
    mean_token_probability: float | None
    minimum_token_probability: float | None
    mean_token_entropy: float | None

    runtime_seconds: float
    peak_cuda_memory_bytes: int | None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("screening configuration must be a mapping")

    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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

            records.append(payload)

    return records


def write_jsonl(path: Path, payloads: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_numeric_answer(text: str) -> str | None:
    """Parse a strict final marker or a numeric-only response."""

    final_matches = _FINAL_PATTERN.findall(text)

    if final_matches:
        candidate = final_matches[-1].replace(",", "")

        try:
            return str(int(candidate))
        except ValueError:
            return None

    stripped = text.strip()

    if not re.fullmatch(r"[-+]?\d[\d,]*", stripped):
        return None

    try:
        return str(int(stripped.replace(",", "")))
    except ValueError:
        return None

def is_prediction_correct(
    prediction: str | None,
    gold_answers: Iterable[str],
) -> bool:
    if prediction is None:
        return False

    normalized_gold = {
        normalized
        for answer in gold_answers
        if (normalized := normalize_numeric_answer(str(answer))) is not None
    }

    return prediction in normalized_gold


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the model chat template when supported."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful arithmetic assistant. Follow the user's "
                "instructions exactly."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    apply_template = getattr(tokenizer, "apply_chat_template", None)

    if callable(apply_template):
        return apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        "System: You are a careful arithmetic assistant.\n"
        f"User: {prompt}\n"
        "Assistant:"
    )


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    try:
        return mapping[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_generation_metrics(
    *,
    generated_token_ids: torch.Tensor,
    generation_scores: tuple[torch.Tensor, ...],
) -> dict[str, float | int | None]:
    """Compute selected-token probabilities and predictive entropy."""

    token_count = int(generated_token_ids.numel())

    if token_count == 0 or not generation_scores:
        return {
            "generated_token_count": token_count,
            "sequence_log_probability": None,
            "mean_token_log_probability": None,
            "mean_token_probability": None,
            "minimum_token_probability": None,
            "mean_token_entropy": None,
        }

    selected_log_probabilities: list[float] = []
    selected_probabilities: list[float] = []
    entropies: list[float] = []

    usable_steps = min(token_count, len(generation_scores))

    for step in range(usable_steps):
        logits = generation_scores[step][0].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

        token_id = int(generated_token_ids[step].item())
        selected_log_probability = float(
            log_probabilities[token_id].item()
        )
        selected_probability = float(
            probabilities[token_id].item()
        )

        entropy = float(
            -(probabilities * log_probabilities).sum().item()
        )

        selected_log_probabilities.append(selected_log_probability)
        selected_probabilities.append(selected_probability)
        entropies.append(entropy)

    if not selected_log_probabilities:
        return {
            "generated_token_count": token_count,
            "sequence_log_probability": None,
            "mean_token_log_probability": None,
            "mean_token_probability": None,
            "minimum_token_probability": None,
            "mean_token_entropy": None,
        }

    return {
        "generated_token_count": token_count,
        "sequence_log_probability": float(
            sum(selected_log_probabilities)
        ),
        "mean_token_log_probability": float(
            sum(selected_log_probabilities)
            / len(selected_log_probabilities)
        ),
        "mean_token_probability": float(
            sum(selected_probabilities)
            / len(selected_probabilities)
        ),
        "minimum_token_probability": float(
            min(selected_probabilities)
        ),
        "mean_token_entropy": float(
            sum(entropies) / len(entropies)
        ),
    }


@torch.inference_mode()
def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    gold_answers: Iterable[str],
    generation_config: Mapping[str, Any],
) -> GenerationMetrics:
    formatted_prompt = build_chat_prompt(tokenizer, prompt)

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    model_device = next(model.parameters()).device
    encoded = {
        key: value.to(model_device)
        for key, value in encoded.items()
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()

    output = model.generate(
        **encoded,
        max_new_tokens=int(generation_config["max_new_tokens"]),
        do_sample=bool(generation_config.get("do_sample", False)),
        use_cache=bool(generation_config.get("use_cache", True)),
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime_seconds = time.perf_counter() - started

    prompt_length = int(encoded["input_ids"].shape[1])
    generated_token_ids = output.sequences[0, prompt_length:]

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    ).strip()

    prediction = normalize_numeric_answer(generated_text)

    confidence = compute_generation_metrics(
        generated_token_ids=generated_token_ids,
        generation_scores=tuple(output.scores),
    )

    peak_memory = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None
    )

    return GenerationMetrics(
        generated_text=generated_text,
        normalized_prediction=prediction,
        is_correct=is_prediction_correct(
            prediction,
            gold_answers,
        ),
        generated_token_count=int(
            confidence["generated_token_count"]
        ),
        reached_generation_limit=(
            int(confidence["generated_token_count"])
            >= int(generation_config["max_new_tokens"])
        ),
        sequence_log_probability=confidence[
            "sequence_log_probability"
        ],
        mean_token_log_probability=confidence[
            "mean_token_log_probability"
        ],
        mean_token_probability=confidence[
            "mean_token_probability"
        ],
        minimum_token_probability=confidence[
            "minimum_token_probability"
        ],
        mean_token_entropy=confidence["mean_token_entropy"],
        runtime_seconds=runtime_seconds,
        peak_cuda_memory_bytes=peak_memory,
    )


def classify_group(
    variants: Mapping[str, Mapping[str, Any]],
    *,
    minimum_probability_gain: float,
    minimum_log_probability_gain: float,
    high_confidence_threshold: float,
) -> tuple[str, dict[str, float | None]]:
    """Classify the intervention response for one three-variant group."""

    original = variants["original"]
    resolved = variants["resolved"]
    control = variants["irrelevant_control"]

    original_correct = bool(original["is_correct"])
    resolved_correct = bool(resolved["is_correct"])
    control_correct = bool(control["is_correct"])

    probability_gain = safe_difference(
        resolved.get("mean_token_probability"),
        original.get("mean_token_probability"),
    )
    control_probability_gain = safe_difference(
        resolved.get("mean_token_probability"),
        control.get("mean_token_probability"),
    )
    log_probability_gain = safe_difference(
        resolved.get("mean_token_log_probability"),
        original.get("mean_token_log_probability"),
    )

    original_probability = original.get("mean_token_probability")

    if not original_correct and resolved_correct and not control_correct:
        status = "strong_resolving"
    elif resolved_correct and not original_correct:
        status = "behavioral_resolving"
    elif original_correct and not resolved_correct:
        status = "harmful"
    elif (
        original_correct
        and resolved_correct
        and control_correct
        and isinstance(original_probability, (int, float))
        and original_probability >= high_confidence_threshold
    ):
        status = "invalid_easy"
    elif (
        probability_gain is not None
        and probability_gain >= minimum_probability_gain
        and (
            log_probability_gain is None
            or log_probability_gain >= minimum_log_probability_gain
        )
    ):
        status = "confidence_improvement"
    else:
        status = "no_effect"

    return status, {
        "resolved_minus_original_mean_probability": probability_gain,
        "resolved_minus_control_mean_probability": control_probability_gain,
        "resolved_minus_original_mean_log_probability": (
            log_probability_gain
        ),
    }


def safe_difference(
    first: Any,
    second: Any,
) -> float | None:
    if not isinstance(first, (int, float)):
        return None

    if not isinstance(second, (int, float)):
        return None

    if not math.isfinite(float(first)):
        return None

    if not math.isfinite(float(second)):
        return None

    return float(first) - float(second)


def run_screening(config_path: Path, *, overwrite: bool) -> None:
    config = load_yaml(config_path)

    input_path = PROJECT_ROOT / config["input"]["curated_groups"]
    output_path = PROJECT_ROOT / config["output"]["records"]
    summary_path = PROJECT_ROOT / config["output"]["summary"]

    for path in (output_path, summary_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}"
            )

    seed = int(config["experiment"]["seed"])
    set_deterministic_seed(seed)

    groups = load_jsonl(input_path)

    if len(groups) != 10:
        raise ValueError(
            f"expected 10 curated reasoning groups, found {len(groups)}"
        )

    model_config = config["model"]
    generation_config = config["generation"]

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
        torch_dtype=resolve_dtype(model_config["dtype"]),
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=bool(
            model_config.get("trust_remote_code", False)
        ),
    )
    model.eval()

    result_records: list[dict[str, Any]] = []

    for group in groups:
        group_id = str(group["group_id"])
        gold_answers = [
            str(answer)
            for answer in group["gold_answers"]
        ]

        for variant in _VARIANTS:
            variant_payload = group[variant]
            prompt = str(variant_payload["prompt"])

            metrics = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                gold_answers=gold_answers,
                generation_config=generation_config,
            )

            result_records.append(
                {
                    "group_id": group_id,
                    "base_id": str(group["base_id"]),
                    "variant": variant,
                    "source": "reasoning",
                    "dataset_name": str(group["dataset_name"]),
                    "domain": group["provenance"]["domain"],
                    "operation": group["provenance"]["operation"],
                    "expression": group["provenance"]["expression"],
                    "gold_answers": gold_answers,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "model_id": model_config["model_id"],
                    "seed": seed,
                    **asdict(metrics),
                }
            )

            print(
                f"{group_id} | {variant} | "
                f"prediction={metrics.normalized_prediction} | "
                f"correct={metrics.is_correct} | "
                f"mean_p={metrics.mean_token_probability}"
            )

    write_jsonl(output_path, result_records)

    by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for record in result_records:
        by_group[record["group_id"]][record["variant"]] = record

    screening_config = config["screening"]
    group_results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for group_id, variants in sorted(by_group.items()):
        missing = sorted(set(_VARIANTS).difference(variants))
        if missing:
            raise ValueError(
                f"group {group_id} is missing variants: {missing}"
            )

        status, deltas = classify_group(
            variants,
            minimum_probability_gain=float(
                screening_config["minimum_probability_gain"]
            ),
            minimum_log_probability_gain=float(
                screening_config["minimum_log_probability_gain"]
            ),
            high_confidence_threshold=float(
                screening_config[
                    "reject_if_original_correct_and_mean_probability_at_least"
                ]
            ),
        )

        status_counts[status] += 1

        group_results.append(
            {
                "group_id": group_id,
                "status": status,
                "correctness": {
                    variant: bool(variants[variant]["is_correct"])
                    for variant in _VARIANTS
                },
                "mean_token_probability": {
                    variant: variants[variant][
                        "mean_token_probability"
                    ]
                    for variant in _VARIANTS
                },
                **deltas,
            }
        )

    variant_accuracy = {}

    for variant in _VARIANTS:
        matching = [
            record
            for record in result_records
            if record["variant"] == variant
        ]
        variant_accuracy[variant] = (
            sum(bool(record["is_correct"]) for record in matching)
            / len(matching)
        )

    summary = {
        "experiment": config["experiment"],
        "model": model_config,
        "generation": generation_config,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "group_count": len(by_group),
        "record_count": len(result_records),
        "variant_accuracy": variant_accuracy,
        "status_counts": dict(sorted(status_counts.items())),
        "group_results": group_results,
    }

    write_json(summary_path, summary)

    print("\nScreening summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen reasoning interventions with Qwen."
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

    run_screening(
        arguments.config,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
