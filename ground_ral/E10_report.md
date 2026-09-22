# E10 — Eq.(9) term balance: evidence, fix, and re-sweep

Grounder top-1, rerank disabled (replay of stored query_graphs; no extra LLM calls). `none` = original lambda_u-weighted fusion; `norm` = node/relation terms min-max normalized across candidates before lambda_u weighting.

## A. Term-scale diagnosis (imbalance proof)

| run | queries | states | median s_node | median s_rel | gap | s_node≥0.9 | s_rel≤0.5 |
|---|---|---|---|---|---|---|---|
| nr3d_deepseek | 99 | 428 | 0.970 | 0.458 | 0.513 | 66.8% | 55.3% |
| nr3d_glm | 100 | 449 | 0.920 | 0.403 | 0.517 | 55.5% | 57.1% |
| nr3d_qwen3 | 100 | 424 | 0.951 | 0.371 | 0.579 | 64.6% | 62.6% |
| nr3d_qwen3_smoke | 100 | 424 | 0.951 | 0.371 | 0.579 | 64.6% | 62.6% |

## B. NR3D lambda_u sweep (100 queries)

| λu | MeanDist (none) | Acc@0.3 (none) | Acc@0.5 (none) | MeanDist (norm) | Acc@0.3 (norm) | Acc@0.5 (norm) |
|---|---|---|---|---|---|---|
| 0.5 | 1.2048 | 0.5200 | 0.5400 | 1.1976 | 0.5100 | 0.5300 |

### C. Decision-change rate (norm vs none, same λu)

| λu | changed top-1 | total | rate |
|---|---|---|---|
| 0.5 | 9 | 100 | 9.0% |

## B. SR3D lambda_u sweep

_not run (missing full-scale NS output)_

## D. Backbone robustness at λu = 0.5 (NR3D)

| backbone | Acc@0.5 (none) | Acc@0.5 (norm) | Acc@0.3 (none) | Acc@0.3 (norm) |
|---|---|---|---|---|
| qwen3 | 0.5400 | 0.5300 | 0.5200 | 0.5100 |
| deepseek | 0.5758 | 0.5657 | 0.5354 | 0.5253 |
| glm | 0.5400 | 0.5400 | 0.5200 | 0.5100 |

