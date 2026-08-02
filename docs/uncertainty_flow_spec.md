# Uncertainty Flow: Scientific Specification

## Central research question

Can internal feature dynamics distinguish different sources of uncertainty and
predict which intervention will resolve them?

## Core hypothesis

Large-language-model uncertainty is not a single scalar property. Different
failure causes produce distinguishable internal token- and layer-wise dynamics.

A source-aware uncertainty estimator should therefore support two tasks:

1. identify why the model is uncertain;
2. select the intervention most likely to improve the answer.

## Initial uncertainty sources

### Low uncertainty

The prompt is sufficiently specified, the relevant information is available,
and the required reasoning is within the model's demonstrated capability.

Expected default action: `answer`.

### Knowledge uncertainty

The question is clear, but the model lacks reliable factual knowledge needed to
answer it.

Examples include obscure facts, rare entities, unsupported claims, and facts
outside the model's effective knowledge boundary.

Expected resolving action: `retrieve`.

### Input ambiguity

The prompt does not contain enough information to determine one intended
interpretation or answer.

Examples include unresolved references, missing entities, missing time
constraints, and several simultaneously valid interpretations.

Expected resolving action: `clarify`.

### Reasoning uncertainty

The prompt contains sufficient information, but producing the answer requires
an unstable, difficult, or error-prone inference process.

Examples include multi-step arithmetic, compositional logic, distracting
information, and long dependency chains.

Expected resolving action: `reason_more`.

## Initial intervention actions

- `answer`: answer directly without additional intervention;
- `retrieve`: provide relevant external evidence;
- `clarify`: request or supply missing disambiguating information;
- `reason_more`: allocate additional structured reasoning;
- `abstain`: decline when no supported intervention can safely resolve the case.

## Important distinctions

### Source is not correctness

An incorrect answer does not automatically imply high uncertainty, and a
correct answer does not prove low uncertainty.

The source label represents the cause of uncertainty in the original query,
not the correctness of one sampled response.

### Source is not dataset identity

Datasets must not serve as direct uncertainty labels.

For example:

- TriviaQA must not automatically mean knowledge uncertainty;
- AmbigNQ must not automatically mean ambiguity uncertainty;
- GSM8K must not automatically mean reasoning uncertainty.

Controlled counterfactual variants must be constructed within base-question
groups.

### Source and resolving action are separate labels

The source and optimal action are related but not identical.

The first pilot uses mostly single-source cases, but the schema must support
mixed-source cases and alternative valid actions in future work.

## Counterfactual group design

Each base item belongs to one `group_id`. All variants derived from the same
base item must remain in the same train, validation, or test partition.

A typical group contains:

- `original`: unresolved original query;
- `resolved`: query after the intended intervention;
- `irrelevant_control`: matched intervention that should not resolve the source;
- `adversarial_control`: misleading or conflicting intervention when applicable.

## Knowledge family

Typical variants:

1. question without evidence;
2. question with correct relevant evidence;
3. question with irrelevant evidence of similar length;
4. question with contradictory evidence.

Expected result:

- valid evidence should improve correctness and reduce knowledge uncertainty;
- irrelevant evidence should not;
- contradictory evidence may increase uncertainty.

## Ambiguity family

Typical variants:

1. ambiguous query;
2. correctly clarified query;
3. query with additional but non-resolving detail;
4. query representing an alternative valid interpretation.

Expected result:

- valid clarification should reduce ambiguity uncertainty;
- retrieval or longer reasoning should not reliably resolve missing intent.

## Reasoning family

Typical variants:

1. original reasoning problem;
2. structured decomposition;
3. increased reasoning budget;
4. irrelevant factual evidence;
5. distractor-enriched version.

Expected result:

- structured or extended reasoning should improve reasoning cases;
- unrelated retrieval should not reliably improve them.

## Pilot scope

The dry pilot contains:

- 10 knowledge groups;
- 10 ambiguity groups;
- 10 reasoning groups.

Each group initially contains three variants:

- original;
- resolving intervention;
- irrelevant control.

Total dry-pilot inputs: 90.

The expanded pilot will contain approximately:

- 200 knowledge groups;
- 200 ambiguity groups;
- 200 reasoning groups.

## Primary tasks

### Task A: source classification

Predict:

- `low_uncertainty`;
- `knowledge`;
- `ambiguity`;
- `reasoning`.

### Task B: intervention selection

Predict:

- `answer`;
- `retrieve`;
- `clarify`;
- `reason_more`;
- `abstain`.

### Task C: intervention-response estimation

Estimate how each candidate intervention changes:

- correctness;
- uncertainty;
- calibration;
- computational cost.

## Evaluation requirements

### Source classification

- macro-F1;
- balanced accuracy;
- class-wise precision, recall, and F1;
- one-vs-rest AUROC;
- multiclass calibration;
- confusion matrix.

### Uncertainty quality

- AUROC;
- AUPRC;
- Brier score;
- expected calibration error;
- risk-coverage;
- selective accuracy.

### Intervention selection

- action-selection accuracy;
- final task accuracy;
- unnecessary intervention rate;
- unresolved failure rate;
- generated-token cost;
- retrieval-call count;
- cost-adjusted utility.

## Required split controls

- split by `group_id`;
- never split variants from the same base question;
- report cross-dataset transfer;
- control prompt and answer length;
- test prompt paraphrases;
- evaluate within correct-only and incorrect-only subsets;
- compare against dataset-identity and lexical baselines.

## Initial feature families

### Probability and length

- sequence likelihood;
- minimum token probability;
- mean and maximum entropy;
- probability margin;
- prompt length;
- generated-answer length.

### Token dynamics

- top-token flip count across layers;
- early-to-late probability change;
- stabilization depth;
- late token correction;
- entropy trajectory statistics.

### Depth dynamics

- adjacent-layer representation change;
- cross-layer convergence;
- late-layer correction magnitude;
- cumulative representation path length;
- convergence reversal count;
- early, middle, and late instability.

### Stage-specific dynamics

- query-stage features before answer generation;
- generation-stage features;
- final-answer features.

## Go/no-go criteria

Proceed beyond the pilot only if:

1. internal dynamics improve source macro-F1 over probability-plus-length;
2. source separation transfers across datasets;
3. the intended intervention produces the largest correctness improvement;
4. internal features add value beyond dataset and lexical cues;
5. source-aware routing beats scalar-confidence routing at matched cost.

## Non-goals for the first pilot

The initial study will not include:

- full autonomous agents;
- long-form claim localization;
- multilingual uncertainty;
- end-to-end LLM fine-tuning;
- large neural routing architectures;
- irreversible real-world actions.
