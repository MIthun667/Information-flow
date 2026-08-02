# IFI repository status

## Outcome

The completed Qwen2.5-1.5B experiment was preserved before extension work.
No model inference or dataset recollection was performed during this audit.
The valid arithmetic and SQuAD collections were not modified or recollected.

## Version 1 preservation

- Snapshot: `outputs/versions/qwen_1_5b_v1/`
- Snapshot size: 175 MB
- Checksum manifest: `outputs/versions/qwen_1_5b_v1_checksums.sha256`
- Files checksummed and verified: 22,839
- Checksum failures: 0
- Writable files remaining in the snapshot after preservation: 0
- Original live collection remains at `outputs/qwen_1_5b/`

Verification command:

```bash
sha256sum -c outputs/versions/qwen_1_5b_v1_checksums.sha256
```

## Version 1 consolidated report

Generated files:

- `reports/version_1/dataset_report.json`
- `reports/version_1/dataset_report.md`

The report contains dataset-level sample counts, correct/incorrect/unresolved
class counts, truncation counts and rates, AUROC, AUPRC, AURC, reliability
status, and AUROC gain over probability-plus-length.

Report checksums:

```text
0c2685eefa734b0752cafb35a4a229610ae846ef2704e0a80aee61f5616f01ed  reports/version_1/dataset_report.json
92978a80f9c83461a91e31618ea592314ff35da34c77b5b23e622bfb3eba932b  reports/version_1/dataset_report.md
```

Regeneration command (refuses to overwrite an existing report):

```bash
./run.sh report-v1
```

## Implemented extensions

### Clean and robust analysis

`src/usig/experiment/extended_analysis.py` implements:

- non-truncated-only filtering for every collection;
- diagnostic-only clean summaries for TruthfulQA;
- five fixed split seeds: 2026, 2027, 2028, 2029, and 2030;
- token-dynamics-only, depth-dynamics-only, and joint IFI ablations;
- probability-plus-length as the comparison baseline;
- paired, record-level bootstrap confidence intervals for AUROC differences;
- 2,000 requested paired bootstrap draws with valid-draw counts recorded;
- strict, alias-aware, and verified TriviaQA label variants.

Clean-analysis outputs are written separately under
`outputs/qwen_1_5b_extended/`; Version 1 metrics are not overwritten. Version
2 clean artifacts use names such as `clean_strict_v2.json`. Version 1 clean
artifacts remain preserved after an audit found that unique SQuAD strata made
their five nominal seeds share one split.

Commands:

```bash
./run.sh clean-analysis
./run.sh trivia-labels
```

These CPU analyses were not executed during this code audit. The commands are
ready for the user to run.

### TriviaQA labels

- `strict`: the original normalized exact-match label;
- `alias`: concise alias-aware matching, including controlled suffix and
  parenthetical rules;
- `verified`: alias match plus normalized token overlap of at least 0.8.

Each label variant is emitted as a separate analysis artifact.

### GSM8K calibration gate

The current isolated calibration writes to
`outputs/qwen_1_5b_calibration_v3/gsm8k/` and does not touch the Version 1
GSM8K collection. It uses:

- exactly 100 deterministic manifest records;
- 256 maximum new tokens;
- explicit final-answer parsing;
- a maximum 5% truncation rate;
- zero permitted final-answer parsing failures.

The GSM8K final-answer parser recognizes explicitly marked answers, boxed
answers, and the final numeric span. The strict arithmetic parser continues
to reject unmarked ambiguous multiple-number responses.

Command:

```bash
./run.sh calibration
```

This GPU calibration was not run during the audit.

### Configuration

`config/experiments/qwen_1_5b_compact.yaml` now controls:

- generation limits by dataset;
- prompt template versions by dataset;
- the global prompt configuration version;
- GSM8K calibration sample count, token limit, parsing requirement, and
  truncation threshold.

The collector records these values in experiment identity and metadata and
rejects prompt-template version mismatches.

### Repair and dataset reruns

Commands:

```bash
./run.sh repair
./run.sh rerun triviaqa
./run.sh rerun ambignq
./run.sh rerun gsm8k_calibration
./run.sh rerun truthfulqa
```

Repair outputs go to `outputs/qwen_1_5b_repairs/`. The runner explicitly
refuses reruns of:

- `ifi_arith_source`
- `ifi_arith_larger_integer`
- `ifi_arith_moderate_multiplicative`
- `squad`

`DRY_RUN=1` is supported by calibration and dataset-specific reruns.

## Generated and modified files

Generated:

- `outputs/versions/qwen_1_5b_v1/` (22,839 preserved files)
- `outputs/versions/qwen_1_5b_v1_checksums.sha256`
- `reports/version_1/dataset_report.json`
- `reports/version_1/dataset_report.md`
- `src/usig/experiment/extended_analysis.py`
- `tests/test_extended_analysis.py`
- `PROJECT_STATUS.md`

Modified:

- `config/experiments/qwen_1_5b_compact.yaml`
- `run.sh`
- `src/usig/evaluation/arithmetic.py`
- `src/usig/experiment/compact_analysis.py`
- `src/usig/experiment/generation.py`
- `src/usig/experiment/large_collection.py`
- `tests/test_ifi_arith.py`
- `tests/test_large_experiment.py`
- `tests/test_run_shell.py`

## Verification results

Syntax and import-bytecode checks:

```bash
bash -n run.sh
/home/mithun-hossain/Desktop/myenv/bin/python -m compileall -q src
```

Both completed successfully.

Test collection: 180 tests.

Exact per-file results:

```text
tests/test_experiment.py          38 passed
tests/test_foundation.py          25 passed
tests/test_ifi_arith.py           41 passed
tests/test_large_experiment.py    28 passed
tests/test_pilot_collection.py    23 passed
tests/test_run_shell.py           21 passed
tests/test_extended_analysis.py    4 passed
TOTAL                            180 passed
```

Warnings were limited to pre-existing SWIG deprecation warnings in the
environment. No test failures occurred.

Runner dry-run verification:

```bash
DRY_RUN=1 ./run.sh calibration
DRY_RUN=1 ./run.sh rerun triviaqa
DRY_RUN=1 ./run.sh rerun squad
```

The first two produced isolated collection and verification commands. The
SQuAD rerun was rejected with exit status 2, as required.

## Recommended execution order

```bash
./run.sh clean-analysis
./run.sh trivia-labels
./run.sh calibration
```

Run `./run.sh repair` only after reviewing the clean-analysis and calibration
results. It performs new GPU inference for TriviaQA and AmbigNQ but does not
touch the preserved Version 1 outputs.

## Version 3 dataset repairs

Version 3 work is isolated under `outputs/qwen_1_5b_repairs_v3/`. No valid
arithmetic, SQuAD, TriviaQA, Version 1, or Version 2 artifact was modified or
recollected.

### AmbigNQ interpretation-aware labels

Existing generations were re-evaluated before any recollection was permitted.
No AmbigNQ inference was run.

Generated artifacts:

- `outputs/qwen_1_5b_repairs_v3/ambignq_labels/interpretation_labels.jsonl`
- `outputs/qwen_1_5b_repairs_v3/ambignq_labels/class_count_report.json`
- `outputs/qwen_1_5b_repairs_v3/ambignq_labels/target_analysis.json`

Class counts:

```text
incorrect          942
partially_correct   50
fully_correct        8
total             1000
```

Target reliability:

- fully wrong versus at least partially correct: 942/58,
  `preliminary`;
- incomplete versus fully complete: 992/8, `descriptive_only`.

For fully-wrong detection, probability-plus-length AUROC was 0.76838 and
probability-plus-length-plus-compact-IFI AUROC was 0.76885. The paired AUROC
difference was 0.00048 with 95% CI [-0.02729, 0.02582]. There is no supported
incremental IFI gain.

The completeness target remains descriptive because it has only eight fully
complete examples.

Commands:

```bash
./run.sh ambignq-labels
./run.sh analyze-repaired
```

Both commands refuse to overwrite their existing outputs.

### TruthfulQA multiple choice

Deterministic MC1-style manifests were prepared from the locally normalized
TruthfulQA source. The best correct answer is evaluated against the incorrect
answer set with deterministic per-question option ordering.

Generated manifests and checksums:

```text
100  outputs/qwen_1_5b_repairs_v3/manifests/truthfulqa_mc_calibration_v3.jsonl
     7f0876ec17a9fb2a7a3d46a32bd2651baf9c688abde2fc684e5609e58160ca8e
790  outputs/qwen_1_5b_repairs_v3/manifests/truthfulqa_mc_full_v3.jsonl
     ac5f62a744f06d10b88f7f6f467a8978500c818978b1e57cfd5ba45ffb43107f
```

The collector records:

- predicted and correct option indices;
- normalized option probabilities and option log probabilities;
- selected-option compact IFI signatures;
- per-option hidden-state signature ablations;
- binary MC correctness;
- checksummed prediction, compact-signature, and ablation collections.

The calibration gate requires exactly 100 resolved records and at least 20
examples in each correctness class. Full MC inference is refused if that gate
is missing or failed.

The generative TruthfulQA lexical diagnostics remain separate in the preserved
collection and are never used as primary MC truth labels.

Commands:

```bash
./run.sh truthfulqa-mc-calibration
./run.sh truthfulqa-mc
./run.sh analyze-repaired
```

Standard compact, residualized, non-truncated clean, five-seed, and paired
bootstrap analyses are run by `analyze-repaired` once the full verified MC
collection exists.

No TruthfulQA MC GPU inference was run during implementation.

### GSM8K calibration Version 4

Generated manifest:

```text
100  outputs/qwen_1_5b_repairs_v3/manifests/gsm8k_calibration_v4_100.jsonl
     367020964eac28e84053cdbdb006169c179d3af4ef3504615822c9f59d4e871e
```

Version 4 uses:

- exactly 100 deterministic examples;
- 512 maximum new tokens;
- the `gsm8k_concise_v4` prompt ending in
  `Final answer: <number>`;
- a stopping criterion only after a complete final-answer line terminated by
  a newline, avoiding premature stopping on a partial number;
- full-trace compact IFI extraction;
- a 32-token final-answer-window IFI ablation.

Diagnostics record EOS status, stop reasons, truncation, parse failures,
answer-marker presence and position, final-line stopping, response length,
token repetition rate, and the ten longest responses.

The gate requires both truncation and parse-failure rates to be no greater
than 5%. There is no full Version 4 collection command; the gate-validation
API refuses full collection authorization unless a valid passing gate exists.

Commands:

```bash
./run.sh gsm8k-calibration-v4
./run.sh gsm8k-diagnostics
```

No Version 4 GPU inference was run during implementation.

### Version 3 runner commands

The following isolated commands are available:

```text
ambignq-labels
gsm8k-diagnostics
gsm8k-calibration-v4
truthfulqa-mc-calibration
truthfulqa-mc
analyze-repaired
report-v3
```

`report-v3` creates a new numbered report on every invocation and never
overwrites an earlier report. The initial CPU-only report is:

- `reports/version_3/repair_report.json`
- `reports/version_3/repair_report.md`

### Version 3 changed files

Added:

- `src/usig/experiment/repair_v3.py`
- `src/usig/experiment/truthfulqa_mc.py`
- `src/usig/experiment/gsm8k_v4.py`
- `tests/test_repairs_v3.py`

Modified:

- `config/experiments/qwen_1_5b_compact.yaml`
- `config/prompts/benchmark_prompts.yaml`
- `run.sh`
- `src/usig/experiment/extended_analysis.py`
- `src/usig/experiment/generation.py`
- `src/usig/experiment/large_collection.py`
- `tests/test_extended_analysis.py`
- `tests/test_run_shell.py`
- `PROJECT_STATUS.md`

Generated:

- the three checksummed manifests listed above;
- AmbigNQ labels, class report, and target analysis;
- the initial Version 3 JSON and Markdown report.

### Version 3 verification

Commands:

```bash
bash -n run.sh
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m compileall -q src
DRY_RUN=1 ./run.sh gsm8k-calibration-v4
DRY_RUN=1 ./run.sh truthfulqa-mc-calibration
DRY_RUN=1 ./run.sh truthfulqa-mc
sha256sum -c outputs/versions/qwen_1_5b_v1_checksums.sha256
```

Results:

- shell syntax: passed;
- Python bytecode compilation: passed;
- all three GPU workflows produced isolated dry-run commands;
- Version 1 checksum verification: 22,839 passed, zero failures;
- writable files in the read-only Version 1 snapshot: zero;
- collected tests: 191.

Exact test counts:

```text
tests/test_experiment.py          38 passed
tests/test_foundation.py          25 passed
tests/test_ifi_arith.py           41 passed
tests/test_large_experiment.py    28 passed
tests/test_pilot_collection.py    23 passed
tests/test_run_shell.py           26 passed
tests/test_extended_analysis.py    4 passed
tests/test_repairs_v3.py           6 passed
TOTAL                            191 passed
```

Warnings were limited to the existing SWIG deprecation warnings.

### Version 3 limitations

- AmbigNQ completeness has only eight fully complete examples.
- Interpretation precision is segment-based and intentionally conservative.
- TruthfulQA MC option scoring and hidden-state collection require GPU
  execution and remain unobserved until calibration runs.
- Very short selected MC options may have undefined scalar IFI and will be
  excluded according to the existing no-imputation policy.
- GSM8K Version 4 may still fail the gate; 512 tokens and safe final-line
  stopping reduce risk but do not guarantee acceptable truncation.
- A passing calibration gate is an execution prerequisite, not evidence that
  IFI improves error detection.

### Required Version 3 execution order

AmbigNQ CPU relabeling and analysis are already complete. Continue with:

```bash
./run.sh truthfulqa-mc-calibration
```

If and only if its gate passes:

```bash
./run.sh truthfulqa-mc
```

Then run the independent GSM8K gate:

```bash
./run.sh gsm8k-calibration-v4
./run.sh gsm8k-diagnostics
```

Finally:

```bash
./run.sh analyze-repaired
./run.sh report-v3
```

## Scientific-validity extension (2026-07-29)

The objective remains to test whether token- and depth-transition signatures
add uncertainty information beyond probability plus length. No code in this
extension selects labels, prompts, thresholds, or IFI variants to improve a
reported benchmark number.

### Preservation and isolation

- Version 1, Version 2, arithmetic, SQuAD, and the original TriviaQA
  collections were not recollected or modified.
- New CPU artifacts are under `outputs/qwen_1_5b_repairs_v3/`.
- Future large-model outputs are isolated under
  `outputs/model_replication/<model_key>/`.
- All artifact writers refuse existing destinations. Full GSM8K and TriviaQA
  extension runners also reject an existing output directory.

### Implemented protocols and files

- `src/usig/experiment/scientific_v3.py`: non-truncated SQuAD depth/token
  ablations, coefficient stability, high-confidence errors, depth profiles,
  TriviaQA power and gated non-overlap manifest construction, and the
  TruthfulQA high-confidence-false subgroup.
- `src/usig/experiment/gsm8k_v4.py`: exact 100-record calibration, four-gram
  and sentence repetition diagnostics, exact-answer accuracy, safe projected
  sample sizing, model-capability rejection, and gated full-manifest creation.
- `src/usig/experiment/repair_v3.py`: unresolved AmbigNQ exclusion, manual
  audit worksheet/status, continuous coverage ranking, calibration metrics,
  and expanded V3 report discovery.
- `src/usig/experiment/truthfulqa_mc.py`: official-style MC1 best-answer and
  MC2 multiple-true-answer fields, probabilities, and answer-token signatures.
- `src/usig/experiment/compact_analysis.py` and
  `src/usig/experiment/extended_analysis.py`: generalized fold-local residual
  IFI, standardized coefficients, Brier score, 10-bin ECE, NLL, fixed-coverage
  risk, and fixed-risk coverage.
- `config/models/qwen2_5_7b.yaml` and `config/models/mistral_7b.yaml`: fixed
  evaluator, normalized layer position, native hidden dimension, calibration,
  and resource-accounting policies.
- `config/prompts/benchmark_prompts.yaml`: checksummed five-shot GSM8K V4 and
  five-shot shortest-answer TriviaQA extension prompts. The preserved
  arithmetic prompt remains separate.

Runner commands now include all requested entries:

```text
ambignq-labels                 gsm8k-diagnostics
gsm8k-calibration-v4           gsm8k-full
truthfulqa-mc-calibration      truthfulqa-mc
trivia-power                   trivia-extend
analyze-repaired               model-calibration <model>
analyze-cross-model            report-v3
```

Supported model keys are exactly `qwen2_5_7b` and `mistral_7b`; other values
fail with status 2. Every inference runner supports `DRY_RUN=1`.

### CPU results generated in this extension

AmbigNQ automatic interpretation labels remain provisional until independent
manual review:

```text
records                         1,000
unresolved                         50
incorrect                         903
partially correct                  39
fully correct                       8
useful-answer minority             47  exploratory
completeness minority                8  descriptive only
```

On the 950 resolved examples, compact IFI added to probability plus length
did not improve the useful-answer AUROC: difference `-0.0110`, paired 95% CI
`[-0.0446, 0.0186]`. For completeness it was worse: `-0.0303`,
`[-0.0538, -0.0072]`, but that target has only eight minority examples and
supports no inferential claim. Uncertainty versus negative interpretation
coverage had Spearman `0.1974` (`p=8.43e-10`); this is provisional pending the
manual audit.

TriviaQA’s official alias-aware clean analysis has 795 usable records
(185 correct, 610 incorrect). The observed token-dynamics gain is `0.00980`
with paired 95% CI `[-0.00057, 0.02019]`. Approximate projected two-sided
power at 2,000 usable records is `0.835`; therefore the prespecified rule
recommends, but does not automatically launch, a non-overlapping extension.

SQuAD used the unchanged Version 2 collection and excluded 397 truncated
records, leaving 1,103 (263 correct, 840 incorrect). Five split checksums are
distinct. Selected prespecified results versus probability plus length
(AUROC `0.8369`) are:

```text
token dynamics                 AUROC 0.8755  gain 0.0386  CI [0.0049, 0.0728]
full depth dynamics            AUROC 0.9249  gain 0.0880  CI [0.0602, 0.1195]
joint token + depth            AUROC 0.9571  gain 0.1202  CI [0.0945, 0.1494]
P + L + residual joint IFI     AUROC 0.9630  gain 0.1261  CI [0.1025, 0.1532]
```

These are SQuAD-specific results, not evidence that IFI works on every
benchmark. The detailed file also contains early/middle/late combinations,
five-seed metrics, standardized coefficient stability, 105 high-confidence
incorrect records, and answerability/depth profiles.

Generated files:

- `outputs/qwen_1_5b_repairs_v3/ambignq_labels_v2/`
  (`interpretation_labels.jsonl`, `class_count_report.json`,
  `manual_audit_sample.jsonl`, `manual_audit_status.json`,
  `target_analysis.json`);
- `outputs/qwen_1_5b_repairs_v3/squad_depth_v3/depth_analysis.json`;
- `outputs/qwen_1_5b_repairs_v3/triviaqa_power_v1/power_analysis.json`;
- a new numbered JSON/Markdown report under `reports/version_3/`.

### Verification

Executed:

```bash
bash -n run.sh
/home/mithun-hossain/Desktop/myenv/bin/python -m compileall -q src
DRY_RUN=1 ./run.sh gsm8k-full
DRY_RUN=1 ./run.sh trivia-extend
DRY_RUN=1 ./run.sh model-calibration qwen2_5_7b
DRY_RUN=1 ./run.sh model-calibration mistral_7b
/home/mithun-hossain/Desktop/myenv/bin/python -m pytest -q
```

Results: shell and compile checks passed; all four inference paths produced
isolated commands; invalid model selection failed; `195 passed`, `0 failed`,
`0 skipped`, and `2` pre-existing SWIG deprecation warnings in `64.73s`.
No GPU inference was executed.

### Remaining limitations and decisions

- Complete `manual_audit_sample.jsonl` before treating AmbigNQ automatic
  labels as accepted.
- TruthfulQA MC calibration and collection have not run; no objective MC1/MC2
  result or high-confidence-false result exists yet.
- GSM8K V4 has not run. The full runner will refuse truncation or parsing over
  5%, loop incidence over 5% using a per-response repetition threshold of
  0.50, accuracy below 5%, a one-class calibration, or projected sample size
  beyond the 1,319 available test records.
- Sampling-based and self-assessment baselines require additional model
  forward passes and remain unexecuted. Existing records provide generation
  latency and probability/transition baselines, but do not contain complete
  peak-memory or semantic-clustering measurements. No comparative cost claim
  should be made until a controlled subset is collected.
- SQuAD cannot add final-layer pooling or final-answer probes without
  prohibited recollection because full hidden tensors were not retained.

Required next review order:

```bash
./run.sh truthfulqa-mc-calibration
./run.sh gsm8k-diagnostics
./run.sh gsm8k-calibration-v4
./run.sh trivia-power
```

After reviewing passing gates and the AmbigNQ audit:

```bash
./run.sh truthfulqa-mc
./run.sh gsm8k-full
./run.sh trivia-extend
./run.sh analyze-repaired
./run.sh report-v3
```

### TruthfulQA MC token-boundary repair (2026-07-29 18:31 +06)

The first `truthfulqa-mc-calibration` attempt stopped after ten records because
joint tokenization of `prompt + option` allowed the prompt's trailing-space
token to merge with a short option. This made a nonempty option appear to have
zero continuation tokens. The ten partial V4 records and their log were
preserved unchanged.

The corrected scorer tokenizes the prompt and option separately, rejects
actually empty options, concatenates their token IDs, and scores only the
option-token positions. Because this changes the token-boundary protocol,
resumption into V4 would be scientifically invalid. The unchanged runner
command now writes a new V5 manifest and destination:

```text
outputs/qwen_1_5b_repairs_v3/manifests/truthfulqa_mc_calibration_v5.jsonl
outputs/qwen_1_5b_repairs_v3/truthfulqa_mc_calibration_v5/
```

The preserved failed attempt remains at
`outputs/qwen_1_5b_repairs_v3/truthfulqa_mc_calibration_v4/` and is not used
by any gate, full collection, or analysis command.

### TriviaQA extension result (2026-07-29 18:41 +06)

The power-authorized extension collected 1,000 records with zero overlap with
the original collection. Verification found 1,000 valid predictions, compact
signatures, and ablation records, with zero missing, unexpected, duplicate,
checksum-failing, corrupt, or non-finite artifacts.

The official alias-aware clean analysis excluded seven truncated records:

```text
usable                              993
alias-aware correct                 324
alias-aware incorrect               669
probability + length AUROC        0.87279
token dynamics AUROC              0.86978  gain -0.00302
token gain paired 95% CI          [-0.00639, 0.00033]
depth dynamics AUROC              0.87270  gain -0.00009
depth gain paired 95% CI          [-0.00509, 0.00487]
joint token + depth AUROC         0.87004  gain -0.00275
joint gain paired 95% CI          [-0.00838, 0.00288]
```

Thus the small positive effect in the original collection did not replicate
under the prespecified five-shot shortest-answer extension protocol. All
confidence intervals include zero. The original and extension are reported
separately because their prompts differ; pooling them as if they shared one
generation protocol would introduce a prompt confound.

The first extension analysis file, `clean_alias_v1.json`, incorrectly retained
strict labels because dataset detection used the isolated directory name. It
is preserved as invalid diagnostic history. Dataset detection was corrected
to use prediction metadata, and the accepted alias-aware result is
`outputs/qwen_1_5b_repairs_v3/triviaqa_extension_v1/clean_alias_v2.json`.

## Qwen2.5-7B primary-model run (2026-07-29)

Primary model: `Qwen/Qwen2.5-7B-Instruct`, cached revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16 on an RTX 5090.
All outputs are isolated under `outputs/models/qwen2_5_7b/`; no 1.5B artifact
was overwritten.

Independent 100-record gates:

```text
source arithmetic       95/5   FAIL projected errors 50/1000
larger-integer          67/33  PASS
moderate arithmetic     98/2   FAIL projected errors 20/1000
SQuAD                    29/71  FAIL truncation 18%
TriviaQA aliases         67/33  PASS, truncation 1%
GSM8K                    86/14  PASS, parse failure 1%, truncation 0%
AmbigNQ useful answer     6/94  FAIL projected useful answers 60/1000
TruthfulQA MC1           22/78  PASS
```

Gate-approved full GPU collections completed:

- larger-integer arithmetic: 1,000;
- GSM8K: 1,319;
- TriviaQA base and zero-overlap extension: 1,000 + 1,000;
- TruthfulQA MC: 790.

Primary clean results:

```text
dataset                   clean N  P+L AUROC  joint AUROC  gain       paired 95% CI
larger-integer              1000     0.95620      0.96234   +0.00614  [-0.00149, 0.01362]
GSM8K                       1319     0.78475      0.80888   +0.02413  [-0.00013, 0.04953]
TriviaQA base aliases        996     0.83370      0.83783   +0.00413  [-0.00906, 0.01760]
TriviaQA extension aliases   996     0.85496      0.84802   -0.00694  [-0.01793, 0.00411]
TruthfulQA MC1               712     0.70700      0.72039   +0.01338  [-0.01227, 0.03941]
```

GSM8K depth dynamics alone produced AUROC `0.81462`, gain `+0.02987`,
paired 95% CI `[0.00627, 0.05460]`; this is the only statistically positive
7B result among the completed prespecified ablations. Joint IFI remains
inconclusive.

During grouped arithmetic execution, a Bash conditional-context bug allowed
791 source-arithmetic records to be generated after its gate rejection. The
process was interrupted, the guard was corrected to return before collection,
and those incomplete isolated records are excluded from analysis and reports.
Moderate arithmetic was not collected.

Verification after execution:

```text
bash syntax passed
Python compilation passed
198 tests passed, 0 failed, 0 skipped, 2 SWIG deprecation warnings
runtime 65.63 seconds
```

## Qwen2.5-7B SQuAD/AmbigNQ protocol repairs

Original calibration audits were preserved under
`outputs/models/qwen2_5_7b/repairs/`. Original SQuAD had 18/100 truncated
responses (48-token limit), 26 correct, and 56 ordinary incorrect. Original
AmbigNQ provisional interpretation labels were 83 incorrect, 4 partial,
1 full, and 12 unresolved.

SQuAD repair `squad_strict_concise_v2` uses a strict answer-only or exact
`UNANSWERABLE` prompt, greedy decoding, 64 tokens, EOS, and safe first-line
stopping. Its isolated calibration produced 100 valid predictions/signatures,
0 truncations, 0 parse failures, 65 correct and 35 incorrect. Projected N is
286 and the repaired SQuAD gate passed; full collection is authorized but was
not launched.

AmbigNQ repair `ambignq_numbered_v3` uses numbered interpretation/answer
blocks, greedy decoding, and 256 tokens. It produced 100 valid records with
0 truncations: 73 incorrect, 16 partial, 1 full, and 10 unresolved under the
provisional interpretation-aware evaluator. Useful-answer balance projects
adequately, but the final gate remains blocked because manual label precision
and disagreement have not been measured. Completeness remains descriptive.

New commands:

```text
audit-model-dataset qwen2_5_7b {squad|ambignq}
repair-calibration qwen2_5_7b {squad|ambignq}
repair-report qwen2_5_7b {squad|ambignq}
```

Final validation: shell syntax and compilation passed; 198 tests passed,
0 failed, 0 skipped, with 2 existing SWIG warnings in 65.81 seconds.
