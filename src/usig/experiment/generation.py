from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import StoppingCriteria, StoppingCriteriaList

DEFAULT_GENERATION_CONFIG = Path("config/experiments/qwen_1_5b_compact.yaml")
# Legacy pilot collector compatibility. New large collections load limits from YAML.
MAX_NEW_TOKENS = {
    "ifi_arith": 24,
    "gsm8k": 512,
    "truthfulqa": 48,
    "triviaqa": 64,
    "ambignq": 96,
    "squad": 48,
}


def load_generation_config(
    project_root: Path, config_path: Path | None = None
) -> dict[str, Any]:
    path = config_path or project_root / DEFAULT_GENERATION_CONFIG
    if not path.is_absolute():
        path = project_root / path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    generation = payload["generation"]
    limits = generation["max_new_tokens"]
    if any(not isinstance(value, int) or value < 1 for value in limits.values()):
        raise ValueError("All generation limits must be positive integers")
    return {
        "path": path,
        "prompt_version": generation["prompt_version"],
        "prompt_versions": generation["prompt_versions"],
        "max_new_tokens": limits,
        "calibration": generation["calibration"],
    }


def render_prompt(
    tokenizer: Any,
    template: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    values = {"question": record["question"], "context": record.get("context")}
    semantic = template["text"].format(**values)
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": semantic}],
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_status = "official_chat_template"
    else:
        rendered = semantic
        chat_status = "semantic_template_only"
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return {
        "semantic_prompt": semantic,
        "rendered_prompt": rendered,
        "rendered_prompt_checksum": hashlib.sha256(rendered.encode()).hexdigest(),
        "chat_template_status": chat_status,
        "prompt_token_count": int(encoded["input_ids"].shape[1]),
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    *,
    max_new_tokens: int,
    stop_on_final_answer_line: bool = False,
    stop_after_first_line: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    prompt_length = int(input_ids.shape[1])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    class FinalAnswerLineCriteria(StoppingCriteria):
        pattern = re.compile(
            r"(?:^|\n)Final answer:\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
            r"(?:\.\d+)?\s*\n",
            re.IGNORECASE,
        )

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0, prompt_length:].detach().cpu().tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            return bool(self.pattern.search(text))

    class FirstAnswerLineCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0, prompt_length:].detach().cpu().tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            stripped = text.strip()
            return stripped.upper() == "UNANSWERABLE" or (
                "\n" in text and bool(text.splitlines()[0].strip())
            )

    criteria = []
    if stop_on_final_answer_line:
        criteria.append(FinalAnswerLineCriteria())
    if stop_after_first_line:
        criteria.append(FirstAnswerLineCriteria())
    stopping = StoppingCriteriaList(criteria) if criteria else None
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    latency = time.perf_counter() - start
    token_ids = output.sequences[0, prompt_length:].detach().cpu().tolist()
    reached_limit = len(token_ids) >= max_new_tokens
    eos_ids = (
        set(tokenizer.eos_token_id)
        if isinstance(tokenizer.eos_token_id, list)
        else {tokenizer.eos_token_id}
    )
    stop_reason = "eos_token" if token_ids and token_ids[-1] in eos_ids else (
        "token_limit" if reached_limit else "generation_stopped"
    )
    return {
        "full_token_ids": output.sequences.detach(),
        "generated_token_ids": token_ids,
        "scores": output.scores,
        "generated_token_count": len(token_ids),
        "raw_response": tokenizer.decode(token_ids, skip_special_tokens=False),
        "response": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "stop_reason": stop_reason,
        "final_answer_stop_enabled": stop_on_final_answer_line,
        "final_answer_stop_detected": (
            stop_on_final_answer_line
            and stop_reason == "generation_stopped"
        ),
        "token_limit_reached": reached_limit,
        "latency_seconds": latency,
        "peak_allocated_gpu_memory": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "total_token_count": prompt_length + len(token_ids),
    }
