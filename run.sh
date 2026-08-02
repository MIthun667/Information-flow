#!/usr/bin/env bash
set -Eeuo pipefail

trap 'status=$?; printf "ERROR: command failed at line %s (status %s): %s\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/home/mithun-hossain/Desktop/myenv/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$PROJECT_ROOT/data/manifests/qwen_1_5b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/qwen_1_5b}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/outputs/logs/qwen_1_5b}"
DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-0}"
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COLLECTIONS=(
  gsm8k_calibration
  ifi_arith_source
  ifi_arith_larger_integer
  ifi_arith_moderate_multiplicative
  squad
  triviaqa
  ambignq
  truthfulqa
)

declare -A EXPECTED_COUNTS=(
  [gsm8k_calibration]=300
  [ifi_arith_source]=1000
  [ifi_arith_larger_integer]=1000
  [ifi_arith_moderate_multiplicative]=1000
  [squad]=1500
  [triviaqa]=1000
  [ambignq]=1000
  [truthfulqa]=790
)

declare -A LOG_NAMES=(
  [gsm8k_calibration]=gsm8k_collection.log
  [ifi_arith_source]=arithmetic_source_collection.log
  [ifi_arith_larger_integer]=arithmetic_larger_integer_collection.log
  [ifi_arith_moderate_multiplicative]=arithmetic_moderate_multiplicative_collection.log
  [squad]=squad_collection.log
  [triviaqa]=triviaqa_collection.log
  [ambignq]=ambignq_collection.log
  [truthfulqa]=truthfulqa_collection.log
)

usage() {
  cat <<'EOF'
Usage: ./run.sh <command>

Commands:
  help              Show this help. CPU-only.
  check             Verify environment, manifests, cache, GPU, and frozen artifacts.
  gsm8k             Collect the GSM8K calibration set. GPU.
  gsm8k-resume      Resume GSM8K calibration. GPU.
  gsm8k-verify      Verify GSM8K artifacts. CPU-only.
  gsm8k-decision    Create GSM8K class-balance decision reports. CPU-only.
  arithmetic        Collect and verify all three arithmetic domains. GPU.
  squad             Collect and verify SQuAD. GPU.
  triviaqa          Collect and verify TriviaQA. GPU.
  ambignq           Collect and verify AmbigNQ. GPU.
  truthfulqa        Collect and verify TruthfulQA lexical diagnostics. GPU.
  collect-all       Calibrate GSM8K, then continue only with CONTINUE_AFTER_GSM8K=1.
  resume-all        Resume in collection order; skips verified outputs. GPU.
  verify-all        Verify every collection that currently exists. CPU-only.
  analyze           Run compact and fold-local residualized analyses. CPU-only.
  transfer          Run arithmetic source-to-shift evaluation. CPU-only.
  report-v1         Build the immutable Version 1 consolidated report. CPU-only.
  clean-analysis    Analyze non-truncated records with five seeds and ablations.
  trivia-labels     Analyze strict, alias-aware, and verified TriviaQA labels.
  calibration       Run the isolated 100-example GSM8K/256-token gate.
  repair            Rerun only truncation-affected QA into a new repair root.
  rerun DATASET     Rerun one non-frozen dataset into a new repair root.
  ambignq-labels    Re-evaluate existing AmbigNQ interpretations. CPU-only.
  gsm8k-diagnostics Report Version 4 GSM8K calibration diagnostics. CPU-only.
  gsm8k-calibration-v4 Run isolated 100-record/512-token GSM8K gate. GPU.
  gsm8k-full       Run the gate-sized GSM8K collection only after V4 passes. GPU.
  truthfulqa-mc-calibration Run isolated 100-record TruthfulQA MC gate. GPU.
  truthfulqa-mc     Run full TruthfulQA MC only after its gate passes. GPU.
  trivia-power      Evaluate whether a non-overlapping extension is justified.
  trivia-extend     Collect the prespecified extension only if power recommends it.
  analyze-repaired  Analyze available Version 3 repaired labels/collections.
  model-calibration MODEL  Calibrate qwen2_5_7b or mistral_7b. GPU.
  model-run MODEL DATASET Run one independently gated model benchmark. GPU.
  analyze-model MODEL Analyze completed model benchmarks. CPU-only.
  analyze-cross-model Analyze available isolated model replications. CPU-only.
  report-qwen-7b    Build the isolated Qwen2.5-7B report. CPU-only.
  audit-model-dataset MODEL DATASET Audit failed SQuAD/AmbigNQ calibration.
  repair-calibration MODEL DATASET Run isolated repaired 100-record calibration.
  repair-report MODEL DATASET Show the repaired audit and gate artifacts.
  report-v3         Consolidate available Version 3 repair reports. CPU-only.
  all               Check, collect, verify, analyze, and transfer. Substantial GPU time.
  status            Print concise collection status. CPU-only.
  clean-partials    List recognized temporary/corrupt record files. CPU-only.

Outputs:
  Collections: outputs/qwen_1_5b/<collection>/
  Logs:        outputs/logs/qwen_1_5b/

Examples:
  MODEL_ID=Qwen/Qwen2.5-3B-Instruct ./run.sh gsm8k
  ./run.sh gsm8k-resume
  CONTINUE_AFTER_GSM8K=1 ./run.sh all
  DRY_RUN=1 CONTINUE_AFTER_GSM8K=1 ./run.sh all

WARNING: `all` may require substantial GPU time and will not continue beyond
GSM8K unless CONTINUE_AFTER_GSM8K=1 is explicitly set.
EOF
}

print_selection() {
  printf 'Python: %s\n' "$PYTHON_BIN"
  printf 'Model: %s\n' "$MODEL_ID"
  printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    printf 'Missing required file: %s\n' "$path" >&2
    return 1
  }
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_logged() {
  local log_path="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$@"
    return 0
  fi
  mkdir -p -- "$(dirname -- "$log_path")"
  set +e
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\nmodel: %s\nstart: %s\n' "$MODEL_ID" "$(date --iso-8601=seconds)"
    "$@"
    status=$?
    printf 'finish: %s\nexit_status: %s\n' "$(date --iso-8601=seconds)" "$status"
    exit "$status"
  } 2>&1 | tee "$log_path"
  status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

static_checks() {
  [[ -x "$PYTHON_BIN" ]] || {
    printf 'Python executable is unavailable: %s\n' "$PYTHON_BIN" >&2
    return 1
  }
  require_file "$MANIFEST_ROOT/collection_index.json"
  require_file "$PROJECT_ROOT/config/prompts/benchmark_prompts.yaml"
  require_file "$PROJECT_ROOT/config/experiments/qwen_1_5b_compact.yaml"
  require_file "$PROJECT_ROOT/config/analysis/compact_confound_controls.yaml"
  local collection
  for collection in "${COLLECTIONS[@]}"; do
    require_file "$MANIFEST_ROOT/${collection}.jsonl"
  done
  "$PYTHON_BIN" -c 'import usig; print("Local usig import: OK")'
  "$PYTHON_BIN" -m usig.data.large_experiment_manifests verify
  if rg -n '^(from|import)[[:space:]]+ifi([[:space:].]|$)' "$PROJECT_ROOT/src"; then
    printf 'Forbidden parent IFI import detected.\n' >&2
    return 1
  fi
  [[ -d "$PROJECT_ROOT/outputs" && -w "$PROJECT_ROOT/outputs" ]] || {
    printf 'Output directory is not writable: %s\n' "$PROJECT_ROOT/outputs" >&2
    return 1
  }
  "$PYTHON_BIN" -c '
from pathlib import Path
import hashlib
root = Path.cwd()
expected = {
    "outputs/predictions/qwen_ifi_66b0032f646fc519.jsonl": "ee131679054b616852d8db5de67d2c36109a0d1a0783e613f7a17f15b6829769",
    "outputs/signatures/qwen_ifi_66b0032f646fc519.jsonl": "7f6050271d1e2d1136783163a44bba0b02c29bb88526dd2b7964cab9db435f9f",
    "data/manifests/pilots/six_benchmark_seed2026_n600.jsonl": "52c5ceeb1707a20f537deeb54e1d24d3f6484f96bd55d8f1ebd70339a3c518c4",
}
for relative, digest in expected.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"Frozen checksum mismatch: {relative}")
print("Frozen Qwen 0.5B checksums: OK")
'
}

gpu_check() {
  "$PYTHON_BIN" -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"bfloat16 support: {torch.cuda.is_bf16_supported()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA device does not support bfloat16")
'
  local cache_name="models--${MODEL_ID//\//--}"
  local snapshot_root="${HOME}/.cache/huggingface/hub/${cache_name}/snapshots"
  if [[ -d "$snapshot_root" ]]; then
    printf 'Model cache: %s\n' "$snapshot_root"
  else
    printf 'Model cache not found locally: %s\n' "$snapshot_root" >&2
    return 1
  fi
}

check_command() {
  print_selection
  static_checks
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN: skipped CUDA and model loading checks.\n'
    return 0
  fi
  mkdir -p -- "$LOG_ROOT"
  gpu_check
  printf 'Environment and integrity checks passed.\n'
}

manifest_for() {
  printf '%s/%s.jsonl' "$MANIFEST_ROOT" "$1"
}

destination_for() {
  printf '%s/%s' "$OUTPUT_ROOT" "$1"
}

verify_one() {
  local collection="$1"
  local manifest
  local destination
  manifest="$(manifest_for "$collection")"
  destination="$(destination_for "$collection")"
  require_file "$manifest"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --model "$MODEL_ID" --manifest "$manifest" \
      --output-destination "$destination"
    return 0
  fi
  run_logged "$LOG_ROOT/${collection}_verification.log" \
    "$PYTHON_BIN" -m usig.experiment.large_collection verify \
    --model "$MODEL_ID" --manifest "$manifest" \
    --output-destination "$destination"
}

is_verified() {
  local collection="$1"
  local destination
  destination="$(destination_for "$collection")"
  [[ -d "$destination" ]] || return 1
  "$PYTHON_BIN" -m usig.experiment.large_collection verify \
    --model "$MODEL_ID" --manifest "$(manifest_for "$collection")" \
    --output-destination "$destination" >/dev/null 2>&1
}

has_corrupt_partials() {
  local destination="$1"
  "$PYTHON_BIN" -c '
import sys
from pathlib import Path
from usig.experiment.large_collection import partial_artifacts
result = partial_artifacts(Path(sys.argv[1]), clean_confirmed=False)
raise SystemExit(0 if result["recognized_partial_count"] else 1)
' "$destination"
}

collect_one() {
  local collection="$1"
  local operation="$2"
  local manifest
  local destination
  manifest="$(manifest_for "$collection")"
  destination="$(destination_for "$collection")"
  require_file "$manifest"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection "$operation" \
      --model "$MODEL_ID" --manifest "$manifest" \
      --output-destination "$destination"
    return 0
  fi
  if is_verified "$collection"; then
    printf '%s: already verified; skipping.\n' "$collection"
    return 0
  fi
  if [[ -d "$destination" ]] && has_corrupt_partials "$destination"; then
    printf '%s contains corrupt or temporary record files; inspect with clean-partials.\n' \
      "$collection" >&2
    return 1
  fi
  if [[ "$operation" == "collect" && -d "$destination" ]]; then
    printf '%s has partial outputs; use the corresponding resume command.\n' \
      "$collection" >&2
    return 1
  fi
  run_logged "$LOG_ROOT/${LOG_NAMES[$collection]}" \
    "$PYTHON_BIN" -m usig.experiment.large_collection "$operation" \
    --model "$MODEL_ID" --manifest "$manifest" \
    --output-destination "$destination"
}

collect_and_verify() {
  local collection="$1"
  local operation="$2"
  collect_one "$collection" "$operation"
  verify_one "$collection"
}

gsm8k_decision() {
  verify_one gsm8k_calibration
  local destination
  destination="$(destination_for gsm8k_calibration)"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.compact_analysis \
      gsm8k-decision --destination "$destination"
    return 0
  fi
  run_logged "$LOG_ROOT/gsm8k_decision.log" \
    "$PYTHON_BIN" -m usig.experiment.compact_analysis \
    gsm8k-decision --destination "$destination"
  local decision="$destination/class_balance_decisions/gsm8k_calibration.json"
  "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
summary = (
    "# GSM8K calibration decision\n\n"
    f"Correct: {data['"'"'correct_count'"'"']}\n\n"
    f"Incorrect: {data['"'"'incorrect_count'"'"']}\n\n"
    f"Decision: `{data['"'"'decision'"'"']}`\n"
)
path = Path(sys.argv[2])
if path.exists():
    if path.read_text(encoding="utf-8") != summary:
        raise SystemExit(f"Existing summary conflicts with current decision: {path}")
else:
    path.write_text(summary, encoding="utf-8")
print(summary)
' "$decision" "$destination/class_balance_decisions/gsm8k_summary.md"
}

continuation_guard() {
  if [[ "${CONTINUE_AFTER_GSM8K:-0}" != "1" ]]; then
    printf '\nStopped cleanly after GSM8K calibration and decision.\n'
    printf 'Review the decision, then continue with:\n'
    printf '  CONTINUE_AFTER_GSM8K=1 ./run.sh all\n'
    return 1
  fi
  return 0
}

remaining_collections() {
  local operation="$1"
  local collection
  for collection in \
    ifi_arith_source \
    ifi_arith_larger_integer \
    ifi_arith_moderate_multiplicative \
    squad triviaqa ambignq truthfulqa; do
    collect_and_verify "$collection" "$operation"
  done
}

collect_all() {
  check_command
  collect_and_verify gsm8k_calibration collect
  gsm8k_decision
  continuation_guard || return 0
  remaining_collections collect
}

resume_all() {
  check_command
  collect_and_verify gsm8k_calibration resume
  gsm8k_decision
  continuation_guard || return 0
  remaining_collections resume
}

verify_all() {
  local found=0
  local collection
  for collection in "${COLLECTIONS[@]}"; do
    if [[ "$DRY_RUN" == "1" || -d "$(destination_for "$collection")" ]]; then
      found=1
      verify_one "$collection"
    else
      printf '%s: no output destination; skipped.\n' "$collection"
    fi
  done
  [[ "$found" == "1" ]] || printf 'No collection outputs exist yet.\n'
}

analysis_one() {
  local collection="$1"
  local action="$2"
  local filename="$3"
  local destination
  destination="$(destination_for "$collection")"
  verify_one "$collection"
  local output="$destination/confound_controlled_metrics/$filename"
  if [[ "$DRY_RUN" != "1" && -f "$output" ]]; then
    printf '%s %s: analysis already exists; skipping.\n' "$collection" "$action"
    return 0
  fi
  run_logged "$LOG_ROOT/${collection}_${action}_analysis.log" \
    "$PYTHON_BIN" -m usig.experiment.compact_analysis "$action" \
    --destination "$destination" \
    --manifest "$(manifest_for "$collection")" \
    --output "$output"
}

analyze_all() {
  local collection
  for collection in "${COLLECTIONS[@]}"; do
    if [[ "$DRY_RUN" != "1" && ! -d "$(destination_for "$collection")" ]]; then
      continue
    fi
    verify_one "$collection"
    if [[ "$collection" == "truthfulqa" ]]; then
      local lexical_output
      lexical_output="$(destination_for "$collection")/confound_controlled_metrics/lexical_diagnostics.json"
      if [[ "$DRY_RUN" == "1" || ! -f "$lexical_output" ]]; then
        run_logged "$LOG_ROOT/truthfulqa_lexical_analysis.log" \
          "$PYTHON_BIN" -m usig.experiment.compact_analysis lexical \
          --destination "$(destination_for "$collection")" \
          --output "$lexical_output"
      fi
      continue
    fi
    analysis_one "$collection" compact compact_comparisons.json
    analysis_one "$collection" residualized residualized_comparisons.json
    if [[ "$collection" == "squad" ]]; then
      local subset
      for subset in answerable unanswerable; do
        local subset_output
        subset_output="$(destination_for squad)/confound_controlled_metrics/${subset}_comparisons.json"
        if [[ "$DRY_RUN" == "1" || ! -f "$subset_output" ]]; then
          run_logged "$LOG_ROOT/squad_${subset}_analysis.log" \
            "$PYTHON_BIN" -m usig.experiment.compact_analysis compact \
            --destination "$(destination_for squad)" \
            --manifest "$(manifest_for squad)" --subset "$subset" \
            --output "$subset_output"
        fi
      done
    fi
  done
  if [[ "$DRY_RUN" == "1" || -d "$(destination_for ifi_arith_source)" ]]; then
    local arithmetic_output
    arithmetic_output="$(destination_for ifi_arith_source)/confound_controlled_metrics/arithmetic_protocols.json"
    if [[ "$DRY_RUN" == "1" || ! -f "$arithmetic_output" ]]; then
      run_logged "$LOG_ROOT/arithmetic_protocol_analysis.log" \
        "$PYTHON_BIN" -m usig.experiment.compact_analysis arithmetic \
        --destination "$(destination_for ifi_arith_source)" \
        --manifest "$(manifest_for ifi_arith_source)" \
        --output "$arithmetic_output"
    fi
  fi
}

transfer_analysis() {
  local collection
  for collection in \
    ifi_arith_source \
    ifi_arith_larger_integer \
    ifi_arith_moderate_multiplicative; do
    if [[ "$DRY_RUN" != "1" ]]; then
      verify_one "$collection"
    elif [[ ! -f "$(manifest_for "$collection")" ]]; then
      printf 'Missing arithmetic manifest: %s\n' "$(manifest_for "$collection")" >&2
      return 1
    fi
  done
  local output
  output="$(destination_for ifi_arith_source)/arithmetic_transfer_metrics/source_to_shifts.json"
  if [[ "$DRY_RUN" != "1" && -f "$output" ]]; then
    printf 'Arithmetic transfer analysis already exists; skipping.\n'
    return 0
  fi
  run_logged "$LOG_ROOT/arithmetic_transfer.log" \
    "$PYTHON_BIN" -m usig.experiment.compact_analysis transfer \
    --source-destination "$(destination_for ifi_arith_source)" \
    --shift-destination "$(destination_for ifi_arith_larger_integer)" \
    --shift-destination "$(destination_for ifi_arith_moderate_multiplicative)" \
    --output "$output"
}

status_command() {
  printf '%-36s %8s %12s %12s %-14s %-10s %s\n' \
    collection expected predictions signatures verification analysis log
  local collection
  for collection in "${COLLECTIONS[@]}"; do
    local destination
    destination="$(destination_for "$collection")"
    local predictions=0
    local signatures=0
    local verification="not-started"
    local analysis="absent"
    if [[ -d "$destination/predictions/records" ]]; then
      predictions="$(find "$destination/predictions/records" -maxdepth 1 -type f -name '*.json' | wc -l)"
    fi
    if [[ -d "$destination/compact_signatures/records" ]]; then
      signatures="$(find "$destination/compact_signatures/records" -maxdepth 1 -type f -name '*.json' | wc -l)"
    fi
    if [[ -d "$destination" ]]; then
      if is_verified "$collection"; then
        verification="verified"
      else
        verification="incomplete"
      fi
    fi
    if [[ -d "$destination/confound_controlled_metrics" ]]; then
      analysis="present"
    fi
    printf '%-36s %8s %12s %12s %-14s %-10s %s\n' \
      "$collection" "${EXPECTED_COUNTS[$collection]}" "$predictions" "$signatures" \
      "$verification" "$analysis" "$LOG_ROOT/${LOG_NAMES[$collection]}"
  done
}

clean_partials() {
  local clean_flag=()
  if [[ "${CONFIRM_CLEAN:-0}" == "1" ]]; then
    clean_flag=(--clean-confirmed)
  else
    printf 'Inspection only. Set CONFIRM_CLEAN=1 to remove only recognized partial files.\n'
  fi
  local collection
  for collection in "${COLLECTIONS[@]}"; do
    local destination
    destination="$(destination_for "$collection")"
    [[ -d "$destination" ]] || continue
    if [[ "$DRY_RUN" == "1" ]]; then
      print_command "$PYTHON_BIN" -m usig.experiment.large_collection partials \
        --output-destination "$destination" "${clean_flag[@]}"
    else
      "$PYTHON_BIN" -m usig.experiment.large_collection partials \
        --output-destination "$destination" "${clean_flag[@]}"
    fi
  done
}

extended_analysis_one() {
  local collection="$1"
  local label_variant="${2:-strict}"
  local analysis_root="${EXTENDED_ANALYSIS_ROOT:-$PROJECT_ROOT/outputs/qwen_1_5b_extended}"
  local destination
  destination="$(destination_for "$collection")"
  local suffix="clean_${label_variant}_v2.json"
  local output="$analysis_root/$collection/$suffix"
  if [[ -f "$output" ]]; then
    printf '%s: extended analysis already exists; skipping %s.\n' "$collection" "$output"
    return 0
  fi
  run_logged "$LOG_ROOT/${collection}_clean_${label_variant}.log" \
    "$PYTHON_BIN" -m usig.experiment.extended_analysis analyze \
    --destination "$destination" --manifest "$(manifest_for "$collection")" \
    --non-truncated-only --label-variant "$label_variant" --output "$output"
}

clean_analysis() {
  local collection
  for collection in "${COLLECTIONS[@]}"; do
    extended_analysis_one "$collection" strict
  done
}

trivia_label_analysis() {
  local variant
  for variant in strict alias verified; do
    extended_analysis_one triviaqa "$variant"
  done
}

report_v1() {
  local source="$PROJECT_ROOT/outputs/versions/qwen_1_5b_v1"
  local report_root="$PROJECT_ROOT/reports/version_1"
  require_file "$PROJECT_ROOT/outputs/versions/qwen_1_5b_v1_checksums.sha256"
  if [[ -f "$report_root/dataset_report.json" ]]; then
    printf 'Version 1 report already exists; skipping.\n'
    return 0
  fi
  "$PYTHON_BIN" -m usig.experiment.extended_analysis report \
    --source-root "$source" \
    --output-json "$report_root/dataset_report.json" \
    --output-markdown "$report_root/dataset_report.md"
}

prepare_calibration_manifest() {
  local target="$MANIFEST_ROOT/gsm8k_calibration_v3_100.jsonl"
  if [[ ! -f "$target" ]]; then
    "$PYTHON_BIN" -c '
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
rows = source.read_text(encoding="utf-8").splitlines()
if len(rows) < 100:
    raise SystemExit("GSM8K manifest has fewer than 100 records")
target.write_text("\n".join(rows[:100]) + "\n", encoding="utf-8")
' "$(manifest_for gsm8k_calibration)" "$target"
  fi
  printf '%s' "$target"
}

calibration_run() {
  local manifest
  if [[ "$DRY_RUN" == "1" ]]; then
    manifest="$MANIFEST_ROOT/gsm8k_calibration_v3_100.jsonl"
  else
    manifest="$(prepare_calibration_manifest)"
  fi
  local destination="$PROJECT_ROOT/outputs/qwen_1_5b_calibration_v3/gsm8k"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
    print_command "$PYTHON_BIN" -m usig.experiment.extended_analysis calibration-gate \
      --destination "$destination" \
      --output "$destination/class_balance_decisions/calibration_gate.json"
    return 0
  fi
  if [[ ! -f "$destination/verification_reports/artifact_checksums.json" ]]; then
    run_logged "$LOG_ROOT/gsm8k_calibration_v3_collection.log" \
      "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
    run_logged "$LOG_ROOT/gsm8k_calibration_v3_verification.log" \
      "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
  fi
  local gate="$destination/class_balance_decisions/calibration_gate.json"
  if [[ ! -f "$gate" ]]; then
    "$PYTHON_BIN" -m usig.experiment.extended_analysis calibration-gate \
      --destination "$destination" --output "$gate"
  fi
  "$PYTHON_BIN" -c '
import json, sys
result = json.load(open(sys.argv[1]))
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
' "$gate"
}

rerun_dataset() {
  local collection="${1:-}"
  case "$collection" in
    gsm8k_calibration|triviaqa|ambignq|truthfulqa) ;;
    ifi_arith_source|ifi_arith_larger_integer|ifi_arith_moderate_multiplicative|squad)
      printf 'Refusing to recollect valid protected dataset: %s\n' "$collection" >&2
      return 2
      ;;
    *)
      printf 'Unsupported rerun dataset: %s\n' "$collection" >&2
      return 2
      ;;
  esac
  local repair_root="${REPAIR_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/qwen_1_5b_repairs}"
  local destination="$repair_root/$collection"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$(manifest_for "$collection")" \
      --output-destination "$destination"
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --model "$MODEL_ID" --manifest "$(manifest_for "$collection")" \
      --output-destination "$destination"
    return 0
  fi
  run_logged "$LOG_ROOT/${collection}_repair_collection.log" \
    "$PYTHON_BIN" -m usig.experiment.large_collection collect \
    --model "$MODEL_ID" --manifest "$(manifest_for "$collection")" \
    --output-destination "$destination"
  run_logged "$LOG_ROOT/${collection}_repair_verification.log" \
    "$PYTHON_BIN" -m usig.experiment.large_collection verify \
    --model "$MODEL_ID" --manifest "$(manifest_for "$collection")" \
    --output-destination "$destination"
}

repair_datasets() {
  rerun_dataset triviaqa
  rerun_dataset ambignq
}

repair_v3_root() {
  printf '%s' "${REPAIR_V3_ROOT:-$PROJECT_ROOT/outputs/qwen_1_5b_repairs_v3}"
}

ambignq_labels_v3() {
  local output
  output="$(repair_v3_root)/ambignq_labels_v2"
  if [[ -f "$output/class_count_report.json" ]]; then
    printf 'AmbigNQ Version 3 labels already exist; skipping.\n'
    return 0
  fi
  "$PYTHON_BIN" -m usig.experiment.repair_v3 ambignq-labels \
    --normalized "$PROJECT_ROOT/data/normalized/ambignq/validation.jsonl" \
    --predictions "$OUTPUT_ROOT/ambignq/predictions/collection.jsonl" \
    --output-directory "$output"
}

prepare_gsm8k_v4_manifest() {
  local output
  output="$(repair_v3_root)/manifests/gsm8k_calibration_v4_100.jsonl"
  if [[ ! -f "$output" ]]; then
    "$PYTHON_BIN" -m usig.experiment.gsm8k_v4 manifest \
      --source "$(manifest_for gsm8k_calibration)" --output "$output" >/dev/null
  fi
  printf '%s' "$output"
}

gsm8k_diagnostics_v4() {
  local destination
  destination="$(repair_v3_root)/gsm8k_calibration_v4"
  local predictions="$destination/predictions/collection.jsonl"
  require_file "$predictions"
  local output="$destination/diagnostics.json"
  if [[ -f "$output" ]]; then
    printf 'GSM8K Version 4 diagnostics already exist; showing gate.\n'
    "$PYTHON_BIN" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=2))' "$output"
    return 0
  fi
  "$PYTHON_BIN" -m usig.experiment.gsm8k_v4 diagnostics \
    --predictions "$predictions" --output "$output"
}

gsm8k_calibration_v4() {
  local destination
  destination="$(repair_v3_root)/gsm8k_calibration_v4"
  local manifest
  if [[ "$DRY_RUN" == "1" ]]; then
    manifest="$(repair_v3_root)/manifests/gsm8k_calibration_v4_100.jsonl"
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$manifest" \
      --output-destination "$destination" --stop-on-final-answer-line \
      --final-answer-window-tokens 32
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
    print_command "$PYTHON_BIN" -m usig.experiment.gsm8k_v4 diagnostics \
      --predictions "$destination/predictions/collection.jsonl" \
      --output "$destination/diagnostics.json"
    return 0
  fi
  manifest="$(prepare_gsm8k_v4_manifest)"
  if [[ ! -f "$destination/verification_reports/artifact_checksums.json" ]]; then
    run_logged "$LOG_ROOT/gsm8k_calibration_v4_collection.log" \
      "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$manifest" \
      --output-destination "$destination" --stop-on-final-answer-line \
      --final-answer-window-tokens 32
    run_logged "$LOG_ROOT/gsm8k_calibration_v4_verification.log" \
      "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
  fi
  gsm8k_diagnostics_v4
  "$PYTHON_BIN" -c '
import json,sys
gate=json.load(open(sys.argv[1]))
raise SystemExit(0 if gate["passed"] else 1)
' "$destination/diagnostics.json"
}

gsm8k_full_v4() {
  local root gate manifest destination
  root="$(repair_v3_root)"
  gate="$root/gsm8k_calibration_v4/diagnostics.json"
  manifest="$root/manifests/gsm8k_full_v4.jsonl"
  destination="$root/gsm8k_full_v4"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.gsm8k_v4 full-manifest \
      --normalized "$PROJECT_ROOT/data/normalized/gsm8k/test.jsonl" \
      --gate "$gate" --output "$manifest"
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$manifest" \
      --output-destination "$destination" --stop-on-final-answer-line \
      --final-answer-window-tokens 32
    return 0
  fi
  require_file "$gate"
  "$PYTHON_BIN" -m usig.experiment.gsm8k_v4 require-gate --gate "$gate" >/dev/null
  [[ -e "$destination" ]] && {
    printf 'Refusing to overwrite GSM8K full destination: %s\n' "$destination" >&2
    return 1
  }
  "$PYTHON_BIN" -m usig.experiment.gsm8k_v4 full-manifest \
    --normalized "$PROJECT_ROOT/data/normalized/gsm8k/test.jsonl" \
    --gate "$gate" --output "$manifest"
  run_logged "$LOG_ROOT/gsm8k_full_v4_collection.log" \
    "$PYTHON_BIN" -m usig.experiment.large_collection collect \
    --model "$MODEL_ID" --manifest "$manifest" \
    --output-destination "$destination" --stop-on-final-answer-line \
    --final-answer-window-tokens 32
}

trivia_power_v3() {
  local output
  output="$(repair_v3_root)/triviaqa_power_v1/power_analysis.json"
  if [[ -f "$output" ]]; then
    "$PYTHON_BIN" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=2))' "$output"
    return 0
  fi
  "$PYTHON_BIN" -m usig.experiment.scientific_v3 trivia-power \
    --alias-result "$PROJECT_ROOT/outputs/qwen_1_5b_extended/triviaqa/clean_alias_v2.json" \
    --output "$output"
}

trivia_extend_v3() {
  local root power manifest destination
  root="$(repair_v3_root)"
  power="$root/triviaqa_power_v1/power_analysis.json"
  manifest="$root/manifests/triviaqa_extension_v1.jsonl"
  destination="$root/triviaqa_extension_v1"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.scientific_v3 trivia-extension-manifest \
      --validation "$PROJECT_ROOT/data/normalized/triviaqa/validation.jsonl" \
      --train "$PROJECT_ROOT/data/normalized/triviaqa/train.jsonl" \
      --existing-manifest "$(manifest_for triviaqa)" --power "$power" --output "$manifest"
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect \
      --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
    return 0
  fi
  require_file "$power"
  [[ -e "$destination" ]] && {
    printf 'Refusing to overwrite TriviaQA extension destination: %s\n' "$destination" >&2
    return 1
  }
  "$PYTHON_BIN" -m usig.experiment.scientific_v3 trivia-extension-manifest \
    --validation "$PROJECT_ROOT/data/normalized/triviaqa/validation.jsonl" \
    --train "$PROJECT_ROOT/data/normalized/triviaqa/train.jsonl" \
    --existing-manifest "$(manifest_for triviaqa)" --power "$power" --output "$manifest"
  run_logged "$LOG_ROOT/triviaqa_extension_v1_collection.log" \
    "$PYTHON_BIN" -m usig.experiment.large_collection collect \
    --model "$MODEL_ID" --manifest "$manifest" --output-destination "$destination"
}

model_calibration_v3() {
  local key="$1"
  [[ "$key" == "qwen2_5_7b" ]] || {
    printf 'Invalid model key: %s (expected qwen2_5_7b)\n' "$key" >&2
    return 2
  }
  local name source available
  while read -r name source available; do
    model_calibrate_one "$name" "$source" "$available"
  done <<EOF
ifi_arith_source $MANIFEST_ROOT/ifi_arith_source.jsonl 1000
ifi_arith_larger_integer $MANIFEST_ROOT/ifi_arith_larger_integer.jsonl 1000
ifi_arith_moderate_multiplicative $MANIFEST_ROOT/ifi_arith_moderate_multiplicative.jsonl 1000
squad $MANIFEST_ROOT/squad.jsonl 1500
triviaqa $MANIFEST_ROOT/triviaqa.jsonl 2000
gsm8k $MANIFEST_ROOT/gsm8k_calibration.jsonl 1319
ambignq $MANIFEST_ROOT/ambignq.jsonl 1000
truthfulqa_mc $PROJECT_ROOT/data/normalized/truthfulqa/all.jsonl 790
EOF
}

model_7b_root() {
  printf '%s' "${MODEL_7B_ROOT:-$PROJECT_ROOT/outputs/models/qwen2_5_7b}"
}

model_calibrate_one() {
  local name="$1" source="$2" available="$3"
  local root manifest destination verification gate
  root="$(model_7b_root)"
  manifest="$root/manifests/calibration_${name}_100.jsonl"
  destination="$root/calibration/$name"
  verification="$destination/verification_reports/model_suite_verification.json"
  gate="$destination/calibration_gate.json"
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$name" == "truthfulqa_mc" ]]; then
      print_command "$PYTHON_BIN" -m usig.experiment.repair_v3 truthfulqa-mc-manifest \
        --normalized "$source" --output "$manifest" --limit 100
      print_command "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc collect \
        --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" \
        --normalized "$source" --destination "$destination"
    else
      print_command "$PYTHON_BIN" -m usig.experiment.model_suite calibration-manifest \
        --source "$source" --output "$manifest"
      print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect \
        --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" \
        --output-destination "$destination"
    fi
    return 0
  fi
  if [[ "$name" == "truthfulqa_mc" ]]; then
    if [[ ! -f "$manifest" ]]; then
      "$PYTHON_BIN" -m usig.experiment.repair_v3 truthfulqa-mc-manifest \
        --normalized "$source" --output "$manifest" --limit 100 >/dev/null
    fi
    if [[ ! -f "$destination/verification_reports/artifact_checksums.json" ]]; then
      run_logged "$root/logs/calibration_${name}.log" \
        "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc collect \
        --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" \
        --normalized "$source" --destination "$destination"
    fi
    "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc verify \
      --manifest "$manifest" --destination "$destination" >"$verification"
  else
    if [[ ! -f "$manifest" ]]; then
      "$PYTHON_BIN" -m usig.experiment.model_suite calibration-manifest \
        --source "$source" --output "$manifest" >/dev/null
    fi
    local stop=()
    [[ "$name" == "gsm8k" ]] && stop=(--stop-on-final-answer-line --final-answer-window-tokens 32)
    if [[ ! -f "$destination/verification_reports/artifact_checksums.json" ]]; then
      run_logged "$root/logs/calibration_${name}.log" \
        "$PYTHON_BIN" -m usig.experiment.large_collection collect \
        --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" \
        --output-destination "$destination" "${stop[@]}"
    fi
    "$PYTHON_BIN" -m usig.experiment.large_collection verify \
      --manifest "$manifest" --output-destination "$destination" >"$verification"
  fi
  [[ -f "$gate" ]] || "$PYTHON_BIN" -m usig.experiment.model_suite gate \
    --predictions "$destination/predictions/collection.jsonl" \
    --verification "$verification" --output "$gate" --dataset "$name" \
    --requested-records 100 --full-available-records "$available"
}

model_run_v1() {
  local key="$1" dataset="$2"
  [[ "$key" == "qwen2_5_7b" ]] || { printf 'Invalid model key: %s\n' "$key" >&2; return 2; }
  case "$dataset" in squad|triviaqa|gsm8k|ambignq|truthfulqa_mc|arithmetic) ;; *)
    printf 'Invalid model dataset: %s\n' "$dataset" >&2; return 2;; esac
  local names=("$dataset")
  [[ "$dataset" == "arithmetic" ]] && names=(ifi_arith_source ifi_arith_larger_integer ifi_arith_moderate_multiplicative)
  local name
  for name in "${names[@]}"; do
    if ! model_run_one_7b "$name"; then
      printf '%s: skipped because its independent gate did not pass.\n' "$name" >&2
      [[ "$dataset" == "arithmetic" ]] || return 1
    fi
  done
}

model_run_one_7b() {
  local name="$1" root gate destination manifest normalized
  root="$(model_7b_root)"
  gate="$root/calibration/$name/calibration_gate.json"
  if [[ "$name" == "squad" || "$name" == "ambignq" ]]; then
    local decision="$(repair_root_7b "$name")/calibration/repair_gate_decision.json"
    if [[ -f "$decision" ]]; then
      if [[ "$DRY_RUN" != "1" ]] && ! "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["full_collection_authorized"] else 1)' "$decision"; then
        return 1
      fi
      gate=""
    fi
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ -n "$gate" ]]; then
      require_file "$gate"
      if ! "$PYTHON_BIN" -m usig.experiment.model_suite require-gate --gate "$gate" >/dev/null; then
        return 1
      fi
    fi
  fi
  if [[ "$name" == "truthfulqa_mc" ]]; then
    manifest="$root/manifests/truthfulqa_mc_full.jsonl"
    normalized="$PROJECT_ROOT/data/normalized/truthfulqa/all.jsonl"
    destination="$root/full/truthfulqa_mc"
    if [[ "$DRY_RUN" == "1" ]]; then
      print_command "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" --normalized "$normalized" --destination "$destination"
      return
    fi
    [[ -f "$manifest" ]] || "$PYTHON_BIN" -m usig.experiment.repair_v3 truthfulqa-mc-manifest --normalized "$normalized" --output "$manifest" >/dev/null
    run_logged "$root/logs/full_${name}.log" "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" --normalized "$normalized" --destination "$destination"
    return
  fi
  if [[ "$name" == "gsm8k" ]]; then
    manifest="$root/manifests/gsm8k_full.jsonl"
    normalized="$PROJECT_ROOT/data/normalized/gsm8k/test.jsonl"
    [[ "$DRY_RUN" == "1" ]] || [[ -f "$manifest" ]] || "$PYTHON_BIN" -m usig.experiment.model_suite full-manifest --normalized "$normalized" --output "$manifest" --dataset gsm8k >/dev/null
  else
    manifest="$MANIFEST_ROOT/$name.jsonl"
  fi
  destination="$root/full/$name"
  [[ "$name" == "squad" || "$name" == "ambignq" ]] && destination="$(repair_root_7b "$name")/full"
  local stop=()
  [[ "$name" == "gsm8k" ]] && stop=(--stop-on-final-answer-line --final-answer-window-tokens 32)
  [[ "$name" == "squad" ]] && stop=(--stop-after-first-line)
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" --output-destination "$destination" "${stop[@]}"
  else
    run_logged "$root/logs/full_${name}.log" "$PYTHON_BIN" -m usig.experiment.large_collection collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" --output-destination "$destination" "${stop[@]}"
  fi
  if [[ "$name" == "triviaqa" ]]; then
    local ext_manifest="$PROJECT_ROOT/outputs/qwen_1_5b_repairs_v3/manifests/triviaqa_extension_v1.jsonl"
    local ext_destination="$root/full/triviaqa_extension"
    if [[ "$DRY_RUN" == "1" ]]; then
      print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$ext_manifest" --output-destination "$ext_destination"
    else
      run_logged "$root/logs/full_triviaqa_extension.log" "$PYTHON_BIN" -m usig.experiment.large_collection collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$ext_manifest" --output-destination "$ext_destination"
    fi
  fi
}

analyze_model_v1() {
  local key="$1"
  [[ "$key" == "qwen2_5_7b" ]] || { printf 'Invalid model key: %s\n' "$key" >&2; return 2; }
  local root name destination manifest variant output
  root="$(model_7b_root)"
  for name in ifi_arith_source ifi_arith_larger_integer ifi_arith_moderate_multiplicative squad gsm8k triviaqa; do
    destination="$root/full/$name"
    [[ -f "$destination/predictions/collection.jsonl" ]] || continue
    manifest="$MANIFEST_ROOT/$name.jsonl"
    [[ "$name" == "gsm8k" ]] && manifest="$root/manifests/gsm8k_full.jsonl"
    variant=strict
    [[ "$name" == "triviaqa" ]] && variant=alias
    output="$destination/analysis/clean_${variant}.json"
    [[ -f "$output" ]] || "$PYTHON_BIN" -m usig.experiment.extended_analysis analyze \
      --destination "$destination" --manifest "$manifest" --output "$output" \
      --non-truncated-only --label-variant "$variant"
  done
  destination="$root/full/triviaqa_extension"
  if [[ -f "$destination/predictions/collection.jsonl" && ! -f "$destination/analysis/clean_alias.json" ]]; then
    "$PYTHON_BIN" -m usig.experiment.extended_analysis analyze \
      --destination "$destination" \
      --manifest "$PROJECT_ROOT/outputs/qwen_1_5b_repairs_v3/manifests/triviaqa_extension_v1.jsonl" \
      --output "$destination/analysis/clean_alias.json" --non-truncated-only \
      --label-variant alias
  fi
  destination="$root/full/truthfulqa_mc"
  if [[ -f "$destination/predictions/collection.jsonl" && ! -f "$destination/analysis/clean_mc1.json" ]]; then
    "$PYTHON_BIN" -m usig.experiment.extended_analysis analyze \
      --destination "$destination" --manifest "$root/manifests/truthfulqa_mc_full.jsonl" \
      --output "$destination/analysis/clean_mc1.json" --non-truncated-only
  fi
  destination="$root/full/ambignq"
  if [[ -f "$destination/predictions/collection.jsonl" && ! -f "$destination/analysis/ambignq_labels/class_count_report.json" ]]; then
    "$PYTHON_BIN" -m usig.experiment.repair_v3 ambignq-labels \
      --normalized "$PROJECT_ROOT/data/normalized/ambignq/validation.jsonl" \
      --predictions "$destination/predictions/collection.jsonl" \
      --output-directory "$destination/analysis/ambignq_labels"
    "$PYTHON_BIN" -m usig.experiment.repair_v3 ambignq-analysis \
      --labels "$destination/analysis/ambignq_labels/interpretation_labels.jsonl" \
      --destination "$destination" --manifest "$MANIFEST_ROOT/ambignq.jsonl" \
      --output "$destination/analysis/ambignq_labels/target_analysis.json"
  fi
}

analyze_cross_model_v3() {
  printf 'Cross-model analysis requires completed aligned Qwen2.5-7B datasets; root: %s\n' "$(model_7b_root)"
}

report_qwen_7b() {
  "$PYTHON_BIN" -m usig.experiment.model_report \
    --model-root "$(model_7b_root)" --output-directory "$PROJECT_ROOT/reports/qwen2_5_7b"
}

repair_root_7b() {
  local dataset="$1"
  [[ "$dataset" == "squad" ]] && printf '%s/repairs/squad_v2' "$(model_7b_root)" || printf '%s/repairs/ambignq_v3' "$(model_7b_root)"
}

audit_model_dataset() {
  local key="$1" dataset="$2"
  [[ "$key" == qwen2_5_7b && ( "$dataset" == squad || "$dataset" == ambignq ) ]] || return 2
  local source="$(model_7b_root)/calibration/$dataset"
  local normalized="$PROJECT_ROOT/data/normalized/$dataset/validation.jsonl"
  "$PYTHON_BIN" -m usig.experiment.dataset_repair --dataset "$dataset" \
    --predictions "$source/predictions/collection.jsonl" \
    --normalized "$normalized" --metadata "$source/extraction_metadata/experiment.json" \
    --output "$(repair_root_7b "$dataset")/audit"
}

repair_calibration_7b() {
  local key="$1" dataset="$2"
  [[ "$key" == qwen2_5_7b && ( "$dataset" == squad || "$dataset" == ambignq ) ]] || return 2
  local root manifest destination
  root="$(repair_root_7b "$dataset")"
  manifest="$root/manifests/calibration_100.jsonl"
  destination="$root/calibration"
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.model_suite calibration-manifest --source "$MANIFEST_ROOT/$dataset.jsonl" --output "$manifest"
    print_command "$PYTHON_BIN" -m usig.experiment.large_collection collect --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" --output-destination "$destination"
    return
  fi
  [[ -f "$root/audit/audit_report.json" ]] || audit_model_dataset "$key" "$dataset"
  [[ -f "$manifest" ]] || "$PYTHON_BIN" -m usig.experiment.model_suite calibration-manifest --source "$MANIFEST_ROOT/$dataset.jsonl" --output "$manifest" >/dev/null
  local stop=()
  [[ "$dataset" == squad ]] && stop=(--stop-after-first-line)
  run_logged "$root/calibration.log" "$PYTHON_BIN" -m usig.experiment.large_collection collect \
    --model Qwen/Qwen2.5-7B-Instruct --manifest "$manifest" \
    --output-destination "$destination" "${stop[@]}"
  "$PYTHON_BIN" -m usig.experiment.large_collection verify --manifest "$manifest" \
    --output-destination "$destination" >"$destination/verification_reports/model_suite_verification.json"
  if [[ "$dataset" == ambignq ]]; then
    "$PYTHON_BIN" -m usig.experiment.repair_v3 ambignq-labels \
      --normalized "$PROJECT_ROOT/data/normalized/ambignq/validation.jsonl" \
      --predictions "$destination/predictions/collection.jsonl" \
      --output-directory "$root/labels"
  fi
  "$PYTHON_BIN" -m usig.experiment.model_suite gate \
    --predictions "$destination/predictions/collection.jsonl" \
    --verification "$destination/verification_reports/model_suite_verification.json" \
    --output "$destination/calibration_gate.json" --dataset "$dataset" \
    --requested-records 100 --full-available-records "$([[ "$dataset" == squad ]] && echo 1500 || echo 1000)"
}

repair_report_7b() {
  local key="$1" dataset="$2"
  [[ "$key" == qwen2_5_7b && ( "$dataset" == squad || "$dataset" == ambignq ) ]] || return 2
  find "$(repair_root_7b "$dataset")" -type f \( -name '*report.json' -o -name 'calibration_gate.json' -o -name 'manual_audit_status.json' \) -print
}

prepare_truthfulqa_mc_manifest() {
  local kind="$1"
  local limit=()
  local output
  output="$(repair_v3_root)/manifests/truthfulqa_mc_${kind}_v5.jsonl"
  [[ "$kind" == "calibration" ]] && limit=(--limit 100)
  if [[ ! -f "$output" ]]; then
    "$PYTHON_BIN" -m usig.experiment.repair_v3 truthfulqa-mc-manifest \
      --normalized "$PROJECT_ROOT/data/normalized/truthfulqa/all.jsonl" \
      --output "$output" "${limit[@]}" >/dev/null
  fi
  printf '%s' "$output"
}

truthfulqa_mc_collect() {
  local kind="$1"
  local destination
  destination="$(repair_v3_root)/truthfulqa_mc_${kind}_v5"
  local manifest
  manifest="$(repair_v3_root)/manifests/truthfulqa_mc_${kind}_v5.jsonl"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc collect \
      --model "$MODEL_ID" --manifest "$manifest" \
      --normalized "$PROJECT_ROOT/data/normalized/truthfulqa/all.jsonl" \
      --destination "$destination"
    return 0
  fi
  manifest="$(prepare_truthfulqa_mc_manifest "$kind")"
  if [[ -f "$destination/verification_reports/artifact_checksums.json" ]]; then
    printf 'TruthfulQA MC %s collection already exists; verifying only.\n' "$kind"
    "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc verify \
      --manifest "$manifest" --destination "$destination"
    return 0
  fi
  run_logged "$LOG_ROOT/truthfulqa_mc_${kind}_collection.log" \
    "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc collect \
    --model "$MODEL_ID" --manifest "$manifest" \
    --normalized "$PROJECT_ROOT/data/normalized/truthfulqa/all.jsonl" \
    --destination "$destination"
  run_logged "$LOG_ROOT/truthfulqa_mc_${kind}_verification.log" \
    "$PYTHON_BIN" -m usig.experiment.truthfulqa_mc verify \
    --manifest "$manifest" --destination "$destination"
}

truthfulqa_mc_calibration() {
  truthfulqa_mc_collect calibration
  [[ "$DRY_RUN" == "1" ]] && return 0
  local destination
  destination="$(repair_v3_root)/truthfulqa_mc_calibration_v5"
  local gate="$destination/calibration_gate.json"
  if [[ ! -f "$gate" ]]; then
    "$PYTHON_BIN" -m usig.experiment.repair_v3 gate \
      --predictions "$destination/predictions/collection.jsonl" \
      --expected-count 100 --output "$gate"
  fi
}

truthfulqa_mc_full() {
  local gate
  gate="$(repair_v3_root)/truthfulqa_mc_calibration_v5/calibration_gate.json"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$gate"
    "$PYTHON_BIN" -c '
import json,sys
if not json.load(open(sys.argv[1]))["passed"]:
    raise SystemExit("TruthfulQA MC full inference refused: calibration gate failed")
' "$gate"
  fi
  truthfulqa_mc_collect full
}

analyze_repaired_v3() {
  local root
  root="$(repair_v3_root)"
  ambignq_labels_v3
  if [[ ! -f "$root/ambignq_labels_v2/target_analysis.json" ]]; then
    "$PYTHON_BIN" -m usig.experiment.repair_v3 ambignq-analysis \
      --labels "$root/ambignq_labels_v2/interpretation_labels.jsonl" \
      --destination "$OUTPUT_ROOT/ambignq" --manifest "$(manifest_for ambignq)" \
      --output "$root/ambignq_labels_v2/target_analysis.json"
  fi
  if [[ ! -f "$root/squad_depth_v3/depth_analysis.json" ]]; then
    "$PYTHON_BIN" -m usig.experiment.scientific_v3 squad-depth \
      --destination "$OUTPUT_ROOT/squad" --manifest "$(manifest_for squad)" \
      --output "$root/squad_depth_v3/depth_analysis.json"
  fi
  local mc="$root/truthfulqa_mc_full_v5"
  if [[ -f "$mc/verification_reports/artifact_checksums.json" ]]; then
    local output="$mc/confound_controlled_metrics"
    if [[ ! -f "$output/compact_comparisons.json" ]]; then
      "$PYTHON_BIN" -m usig.experiment.compact_analysis compact \
        --destination "$mc" \
        --manifest "$root/manifests/truthfulqa_mc_full_v5.jsonl" \
        --output "$output/compact_comparisons.json"
    fi
    if [[ ! -f "$output/residualized_comparisons.json" ]]; then
      "$PYTHON_BIN" -m usig.experiment.compact_analysis residualized \
        --destination "$mc" \
        --manifest "$root/manifests/truthfulqa_mc_full_v5.jsonl" \
        --output "$output/residualized_comparisons.json"
    fi
    if [[ ! -f "$output/clean_paired_v3.json" ]]; then
      "$PYTHON_BIN" -m usig.experiment.extended_analysis analyze \
        --destination "$mc" \
        --manifest "$root/manifests/truthfulqa_mc_full_v5.jsonl" \
        --non-truncated-only --output "$output/clean_paired_v3.json"
    fi
    if [[ ! -f "$output/high_confidence_false_v3.json" ]]; then
      "$PYTHON_BIN" -m usig.experiment.scientific_v3 \
        truthfulqa-high-confidence-false --destination "$mc" \
        --manifest "$root/manifests/truthfulqa_mc_full_v5.jsonl" \
        --output "$output/high_confidence_false_v3.json"
    fi
  else
    printf 'TruthfulQA MC full collection is absent; analysis skipped.\n'
  fi
}

report_v3() {
  local root
  root="$(repair_v3_root)"
  local report_root="$PROJECT_ROOT/reports/version_3"
  local index
  index="$(find "$report_root" -maxdepth 1 -type f -name 'repair_report*.json' 2>/dev/null | wc -l)"
  index=$((index + 1))
  local stem
  stem="$(printf 'repair_report_%03d' "$index")"
  "$PYTHON_BIN" -m usig.experiment.repair_v3 report \
    --repair-root "$root" --output-json "$report_root/${stem}.json" \
    --output-markdown "$report_root/${stem}.md"
}

main() {
  cd -- "$PROJECT_ROOT"
  local command="${1:-help}"
  case "$command" in
    help)
      usage
      ;;
    check)
      check_command
      ;;
    gsm8k)
      print_selection
      collect_one gsm8k_calibration collect
      ;;
    gsm8k-resume)
      print_selection
      collect_one gsm8k_calibration resume
      ;;
    gsm8k-verify)
      print_selection
      verify_one gsm8k_calibration
      ;;
    gsm8k-decision)
      print_selection
      gsm8k_decision
      ;;
    arithmetic)
      print_selection
      collect_and_verify ifi_arith_source collect
      collect_and_verify ifi_arith_larger_integer collect
      collect_and_verify ifi_arith_moderate_multiplicative collect
      ;;
    squad|triviaqa|ambignq|truthfulqa)
      print_selection
      collect_and_verify "$command" collect
      ;;
    collect-all)
      collect_all
      ;;
    resume-all)
      resume_all
      ;;
    verify-all)
      print_selection
      verify_all
      ;;
    analyze)
      print_selection
      analyze_all
      ;;
    transfer)
      print_selection
      transfer_analysis
      ;;
    report-v1)
      report_v1
      ;;
    clean-analysis)
      clean_analysis
      ;;
    trivia-labels)
      trivia_label_analysis
      ;;
    calibration)
      calibration_run
      ;;
    repair)
      repair_datasets
      ;;
    rerun)
      rerun_dataset "${2:-}"
      ;;
    ambignq-labels)
      ambignq_labels_v3
      ;;
    gsm8k-diagnostics)
      gsm8k_diagnostics_v4
      ;;
    gsm8k-calibration-v4)
      gsm8k_calibration_v4
      ;;
    gsm8k-full)
      gsm8k_full_v4
      ;;
    truthfulqa-mc-calibration)
      truthfulqa_mc_calibration
      ;;
    truthfulqa-mc)
      truthfulqa_mc_full
      ;;
    trivia-power)
      trivia_power_v3
      ;;
    trivia-extend)
      trivia_extend_v3
      ;;
    analyze-repaired)
      analyze_repaired_v3
      ;;
    model-calibration)
      model_calibration_v3 "${2:-}"
      ;;
    model-run)
      model_run_v1 "${2:-}" "${3:-}"
      ;;
    analyze-model)
      analyze_model_v1 "${2:-}"
      ;;
    analyze-cross-model)
      analyze_cross_model_v3
      ;;
    report-qwen-7b)
      report_qwen_7b
      ;;
    audit-model-dataset)
      audit_model_dataset "${2:-}" "${3:-}"
      ;;
    repair-calibration)
      repair_calibration_7b "${2:-}" "${3:-}"
      ;;
    repair-report)
      repair_report_7b "${2:-}" "${3:-}"
      ;;
    report-v3)
      report_v3
      ;;
    all)
      collect_all
      if [[ "${CONTINUE_AFTER_GSM8K:-0}" == "1" ]]; then
        verify_all
        analyze_all
        transfer_analysis
        status_command
      fi
      ;;
    status)
      status_command
      ;;
    clean-partials)
      clean_partials
      ;;
    *)
      printf 'Unknown command: %s\n\n' "$command" >&2
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
