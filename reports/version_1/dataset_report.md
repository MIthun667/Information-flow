# Qwen2.5-1.5B IFI Version 1 dataset report

| Dataset | N | Correct | Incorrect | Unresolved | Truncation | AUROC | AUPRC | AURC | Reliability | ΔAUROC vs P+L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ambignq | 1000 | 4 | 996 | 0 | 14.3% | 0.9942 | 1.0000 | 0.9782 | descriptive_only | 0.0005 |
| gsm8k_calibration | 300 | 4 | 296 | 0 | 96.3% | 0.6360 | 0.9921 | 0.9718 | descriptive_only | 0.3370 |
| ifi_arith_larger_integer | 1000 | 488 | 512 | 0 | 5.9% | 0.9700 | 0.9735 | 0.1796 | suitable_for_primary_analysis | -0.0016 |
| ifi_arith_moderate_multiplicative | 1000 | 899 | 101 | 0 | 0.9% | 0.9190 | 0.7366 | 0.0176 | suitable_for_primary_analysis | -0.0268 |
| ifi_arith_source | 1000 | 783 | 217 | 0 | 0.9% | 0.9156 | 0.7449 | 0.0468 | suitable_for_primary_analysis | 0.0077 |
| squad | 1500 | 329 | 1171 | 0 | 26.5% | 0.9088 | 0.9584 | 0.5157 | suitable_for_primary_analysis | 0.1073 |
| triviaqa | 1000 | 46 | 954 | 0 | 20.5% | 0.9881 | 0.9994 | 0.8279 | exploratory | -0.0012 |
| truthfulqa | 790 | 0 | 0 | 790 | 42.7% | — | — | — | diagnostic_only | — |
