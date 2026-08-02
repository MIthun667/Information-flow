# Qwen 1.5B compact uncertainty experiment

## Preferred interface

Use the top-level shell interface from the project root:

```bash
./run.sh check
./run.sh gsm8k
./run.sh gsm8k-resume
./run.sh gsm8k-verify
./run.sh gsm8k-decision
CONTINUE_AFTER_GSM8K=1 ./run.sh all
./run.sh status
./run.sh analyze
./run.sh transfer
DRY_RUN=1 CONTINUE_AFTER_GSM8K=1 ./run.sh all
```

Run `./run.sh help` for the complete command list, GPU/CPU classification,
output locations, model overrides, and continuation warning. `DRY_RUN=1`
validates static inputs and prints the orchestration order without loading a
model, querying the GPU, or creating prediction/signature outputs.

## Why this design is compact

The 0.5B pilot was not statistically reliable: GSM8K had three correct
responses, SQuAD had seven, IFI-ARITH had only 28 examples in the smaller
class, and the headline structured estimator used 132 columns for 100 samples.
Scalar IFI was also strongly associated with generated-answer length and
arithmetic operation.

This experiment freezes five probability features, keeps four length variables
as separate confound controls, and uses exactly ten primary IFI variables.
Scalar IFI is identical to token-instability standard deviation under the
current definition, so the latter is excluded. The 32-position profile and
relative transitions are retained only in separate ablation artifacts.

Residualization is fold-local. A ridge model is fitted on each training fold
using probability and length controls, then applied unchanged to its validation
fold. Correctness labels are never used by residualization. Regularization is
selected from the predetermined values 0.1, 1, and 10 using training folds
only.

Arithmetic source evaluation includes operation-specific results, explicit
operation controls, and four leave-one-operation-out directions. Shift
evaluation trains on source records and applies the fitted models without
refitting to larger-integer and moderate-multiplicative records.

TriviaQA uses normalized exact aliases as the primary label; suffix matching
and containment remain diagnostics. AmbigNQ keeps whole-response and segmented
interpretation metrics separate. TruthfulQA lexical tendencies are diagnostic
and never become headline binary uncertainty labels.

Reliability is descriptive below 20 minority examples, exploratory from 20–49,
preliminary from 50–99, and suitable for primary analysis at 100 or more,
subject to confidence intervals.

## Advanced troubleshooting commands

The direct Python commands below are retained for troubleshooting. Normally,
prefer `./run.sh <command>`. Run direct commands from:

```bash
cd /home/mithun-hossain/Desktop/IFI/UncertaintySignature
```

All long-running commands use `set -o pipefail`, `tee`, and resumable
per-record outputs. Re-running `resume` skips checksum-valid records.

### Verify environment, cache, manifests, and frozen checksums

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.data.large_experiment_manifests verify
nvidia-smi
test -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
sha256sum \
  outputs/predictions/qwen_ifi_66b0032f646fc519.jsonl \
  outputs/signatures/qwen_ifi_66b0032f646fc519.jsonl \
  data/manifests/pilots/six_benchmark_seed2026_n600.jsonl
```

### GSM8K calibration

Screen name: `qwen15-gsm-calibration`

```bash
mkdir -p logs/qwen_1_5b
screen -S qwen15-gsm-calibration
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/gsm8k_calibration.jsonl --output-destination outputs/qwen_1_5b/gsm8k_calibration 2>&1 | tee logs/qwen_1_5b/gsm8k_calibration.log'
```

Verify:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.large_collection verify \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --manifest data/manifests/qwen_1_5b/gsm8k_calibration.jsonl \
  --output-destination outputs/qwen_1_5b/gsm8k_calibration
```

Produce the required stopping decision:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.compact_analysis gsm8k-decision \
  --destination outputs/qwen_1_5b/gsm8k_calibration
```

Do not expand GSM8K unless this decision has been reviewed.

### IFI-ARITH source

Screen name: `qwen15-arithmetic-source`

```bash
screen -S qwen15-arithmetic-source
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/ifi_arith_source.jsonl --output-destination outputs/qwen_1_5b/ifi_arith_source 2>&1 | tee logs/qwen_1_5b/ifi_arith_source.log'
```

### IFI-ARITH larger integer

Screen name: `qwen15-arithmetic-larger`

```bash
screen -S qwen15-arithmetic-larger
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/ifi_arith_larger_integer.jsonl --output-destination outputs/qwen_1_5b/ifi_arith_larger_integer 2>&1 | tee logs/qwen_1_5b/ifi_arith_larger_integer.log'
```

### IFI-ARITH moderate multiplicative

Screen name: `qwen15-arithmetic-moderate`

```bash
screen -S qwen15-arithmetic-moderate
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/ifi_arith_moderate_multiplicative.jsonl --output-destination outputs/qwen_1_5b/ifi_arith_moderate_multiplicative 2>&1 | tee logs/qwen_1_5b/ifi_arith_moderate_multiplicative.log'
```

### SQuAD

Screen name: `qwen15-squad`

```bash
screen -S qwen15-squad
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/squad.jsonl --output-destination outputs/qwen_1_5b/squad 2>&1 | tee logs/qwen_1_5b/squad.log'
```

### TriviaQA

Screen name: `qwen15-trivia`

```bash
screen -S qwen15-trivia
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/triviaqa.jsonl --output-destination outputs/qwen_1_5b/triviaqa 2>&1 | tee logs/qwen_1_5b/triviaqa.log'
```

### AmbigNQ

Screen name: `qwen15-ambignq`

```bash
screen -S qwen15-ambignq
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/ambignq.jsonl --output-destination outputs/qwen_1_5b/ambignq 2>&1 | tee logs/qwen_1_5b/ambignq.log'
```

### TruthfulQA

Screen name: `qwen15-truthfulqa`

```bash
screen -S qwen15-truthfulqa
bash -lc 'set -o pipefail; PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.experiment.large_collection resume --model Qwen/Qwen2.5-1.5B-Instruct --manifest data/manifests/qwen_1_5b/truthfulqa.jsonl --output-destination outputs/qwen_1_5b/truthfulqa 2>&1 | tee logs/qwen_1_5b/truthfulqa.log'
```

### Verify every completed artifact

```bash
for collection in ifi_arith_source ifi_arith_larger_integer ifi_arith_moderate_multiplicative gsm8k_calibration squad triviaqa ambignq truthfulqa; do
  PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
    -m usig.experiment.large_collection verify \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --manifest "data/manifests/qwen_1_5b/${collection}.jsonl" \
    --output-destination "outputs/qwen_1_5b/${collection}" || exit 1
done
```

### Compact-feature analysis

Run once per desired collection, changing `collection`:

```bash
collection=ifi_arith_source
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.compact_analysis compact \
  --destination "outputs/qwen_1_5b/${collection}" \
  --manifest "data/manifests/qwen_1_5b/${collection}.jsonl" \
  --output "outputs/qwen_1_5b/${collection}/confound_controlled_metrics/compact_comparisons.json"
```

For SQuAD, run the answerable and unanswerable targets separately:

```bash
for subset in answerable unanswerable; do
  PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
    -m usig.experiment.compact_analysis compact \
    --destination outputs/qwen_1_5b/squad \
    --manifest data/manifests/qwen_1_5b/squad.jsonl \
    --subset "$subset" \
    --output "outputs/qwen_1_5b/squad/confound_controlled_metrics/${subset}_comparisons.json" || exit 1
done
```

Arithmetic operation protocols:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.compact_analysis arithmetic \
  --destination outputs/qwen_1_5b/ifi_arith_source \
  --manifest data/manifests/qwen_1_5b/ifi_arith_source.jsonl \
  --output outputs/qwen_1_5b/ifi_arith_source/confound_controlled_metrics/arithmetic_protocols.json
```

### Residualized confound-controlled analysis

```bash
collection=ifi_arith_source
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.compact_analysis residualized \
  --destination "outputs/qwen_1_5b/${collection}" \
  --manifest "data/manifests/qwen_1_5b/${collection}.jsonl" \
  --output "outputs/qwen_1_5b/${collection}/confound_controlled_metrics/residualized_comparisons.json"
```

Feature diagnostics:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.compact_analysis features \
  --destination outputs/qwen_1_5b/ifi_arith_source \
  --output outputs/qwen_1_5b/ifi_arith_source/verification_reports/feature_diagnostics.json
```

### Arithmetic source-to-shift transfer

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.compact_analysis transfer \
  --source-destination outputs/qwen_1_5b/ifi_arith_source \
  --shift-destination outputs/qwen_1_5b/ifi_arith_larger_integer \
  --shift-destination outputs/qwen_1_5b/ifi_arith_moderate_multiplicative \
  --output outputs/qwen_1_5b/ifi_arith_source/arithmetic_transfer_metrics/source_to_shifts.json
```
