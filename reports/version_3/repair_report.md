# IFI Version 3 repair report

Included report artifacts: 2

## ambignq_labels/class_count_report.json

```json
{
  "class_counts": {
    "fully_correct": 8,
    "incorrect": 942,
    "partially_correct": 50
  },
  "mean_interpretation_coverage": 0.03877936507936508,
  "mean_interpretation_precision": 0.03604285714285715,
  "sample_count": 1000,
  "source": "outputs/qwen_1_5b/ambignq/predictions/collection.jsonl",
  "targets": {
    "fully_wrong_vs_at_least_partially_correct": {
      "negative_count": 58,
      "positive_class": "fully_wrong",
      "positive_count": 942,
      "reliability_status": "preliminary"
    },
    "incomplete_vs_fully_complete": {
      "negative_count": 8,
      "positive_class": "incomplete",
      "positive_count": 992,
      "reliability_status": "descriptive_only"
    }
  },
  "version": "v3"
}
```

## ambignq_labels/target_analysis.json

```json
{
  "targets": {
    "fully_wrong_target": {
      "probability_length_compact_ifi": {
        "auprc": 0.9821328047483855,
        "aurc": 0.8949209309582058,
        "auroc": 0.7688520389486784,
        "correct_count": 58,
        "gain_over_probability_plus_length": {
          "bootstrap_draws_requested": 2000,
          "bootstrap_draws_valid": 2000,
          "difference": 0.00047587671132576226,
          "paired_bootstrap_95_ci": [
            -0.02729199864915593,
            0.025824074328119305
          ]
        },
        "incorrect_count": 942,
        "reliability_status": "preliminary",
        "sample_count": 1000
      },
      "probability_plus_length": {
        "auprc": 0.9826597491573027,
        "aurc": 0.8941465086058822,
        "auroc": 0.7683761622373526,
        "correct_count": 58,
        "incorrect_count": 942,
        "reliability_status": "preliminary",
        "sample_count": 1000
      },
      "split_seeds": [
        2026,
        2027,
        2028,
        2029,
        2030
      ]
    },
    "incomplete_target": {
      "probability_length_compact_ifi": {
        "auprc": 0.9992575961394352,
        "aurc": 0.9742536065009836,
        "auroc": 0.9143145161290323,
        "correct_count": 8,
        "gain_over_probability_plus_length": {
          "bootstrap_draws_requested": 2000,
          "bootstrap_draws_valid": 1997,
          "difference": -0.02230342741935487,
          "paired_bootstrap_95_ci": [
            -0.04394746459570591,
            -0.0018346081637356049
          ]
        },
        "incorrect_count": 992,
        "reliability_status": "descriptive_only",
        "sample_count": 1000
      },
      "probability_plus_length": {
        "auprc": 0.9994593067716058,
        "aurc": 0.9730185794653583,
        "auroc": 0.9366179435483871,
        "correct_count": 8,
        "incorrect_count": 992,
        "reliability_status": "descriptive_only",
        "sample_count": 1000
      },
      "split_seeds": [
        2026,
        2027,
        2028,
        2029,
        2030
      ]
    }
  },
  "version": "v3"
}
```
