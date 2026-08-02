# Uncertainty Signature

This directory is an independent dataset foundation for research on task-dependent
uncertainty signatures. It does not import the parent `ifi` package and contains no
language-model inference or IFI computation.

## Dataset preparation

Run commands from this directory with the active environment:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.data.prepare all
```

For a read-only validation:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python -m usig.data.prepare all --audit-only
```

The source layout is configured in `config/datasets`. Existing normalized outputs
are protected unless `--overwrite` is explicitly supplied. Canonical data, overlap
records, and audit summaries are separate artifacts.

SQuAD 2.0 is retrieved from the official Hugging Face dataset source and then saved
locally. TriviaQA remains disabled until official question, answer, identifier, and
split metadata have been verified; evidence text files are never treated as examples.

No pilot samples are created by dataset preparation.

## IFI-ARITH

IFI-ARITH is this project's controlled synthetic arithmetic benchmark. Its
`source`, `larger_integer`, and `moderate_multiplicative` domains preserve
separate operand and operation regimes rather than forming one undifferentiated
dataset. It supplies the controlled reasoning component of the broader study;
GSM8K will provide complementary human-authored reasoning evaluation.

IFI-ARITH is not an external public benchmark and does not represent all
mathematical reasoning. Model predictions and future IFI signatures belong in
separate output artifacts.

Validate without writing:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.data.prepare_ifi_arith --audit-only
```

Prepare the full benchmark and deterministic 100-example pilot:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.data.prepare_ifi_arith --pilot
```

## Six-benchmark pilot collection

The first collection contains exactly 100 references from each of:

- IFI-ARITH `source`
- official GSM8K `test`
- TruthfulQA `all`
- Wikipedia TriviaQA `validation`
- AmbigNQ light `validation`
- SQuAD 2.0 `validation`

NQ-Open remains normalized for later Natural Questions relationship and
cross-dataset knowledge studies, but TriviaQA is the second knowledge benchmark
in this first 600-example collection.

Wikipedia TriviaQA was selected because its official release contains complete
train and validation questions, answer aliases, stable identifiers, and
resolvable evidence references with a smaller evidence footprint than the web
variant. Its verified validation subset duplicates ordinary validation records
and is not treated as a separate source. The answerless test split is excluded.
TriviaQA use remains subject to the release README's non-commercial research
terms.

Sampling is deterministic with seed 2026. IFI-ARITH is balanced by operation
and source seed; GSM8K spans question-length and answer-magnitude quartiles;
TruthfulQA includes all categories with proportional allocation; TriviaQA is
stratified by alias count; AmbigNQ is balanced 50/50 by ambiguity; and SQuAD is
balanced 50/50 by answerability with context-group sampling.

Eligibility is applied before selection and is recorded separately. TriviaQA
validation questions related to its selected training split are excluded.
AmbigNQ is checked against selected knowledge-training records using normalized
questions and normalized question-answer relations. Canonical records are never
silently removed.

Semantic, model-independent prompt specifications live in
`config/prompts/benchmark_prompts.yaml`. Local lexical evaluators provide
transparent exact-match and token-F1 diagnostics, but they do not replace
semantic judgment where lexical references are incomplete—especially for
TruthfulQA and ambiguity-aware answers.

Canonical benchmark records, pilot manifests, future model predictions, and
future IFI signatures are distinct artifact classes. Manifests reference
canonical IDs and checksums; they do not embed questions or gold answers.

## Qwen uncertainty-signature experiment

The first model experiment is fixed to `Qwen/Qwen2.5-0.5B-Instruct`, greedy
decoding, and the checksum-locked 600-record pilot manifest. It does not
fine-tune the model. Before collection, validate the manifest and referenced
canonical records:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.collect_signatures validate
```

Start collection with `collect`, or safely continue an interrupted collection
with `resume`. Each example is written atomically to a per-record artifact and
validated by checksum before it is skipped. The consolidated prediction and
signature JSONL files are created only when all 600 valid records exist.

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.collect_signatures collect

PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.collect_signatures resume
```

The experiment identifier is derived from model and tokenizer revisions,
manifest and prompt checksums, generation settings, and feature settings.
Predictions, signatures, metadata, metrics, and diagnostics are kept in
separate `outputs` subdirectories. Full hidden-state tensors are never stored.

After a complete collection, create per-benchmark diagnostics, out-of-fold
uncertainty comparisons, bootstrap intervals, selective-prediction curves,
alignment examples, representative examples, and a short Markdown summary:

```bash
PYTHONPATH=src /home/mithun-hossain/Desktop/myenv/bin/python \
  -m usig.experiment.evaluate_collection qwen_ifi_66b0032f646fc519
```

TruthfulQA reference matching is reported separately as a lexical diagnostic;
unmatched answers remain unresolved and are excluded from headline binary
error-detection results.
