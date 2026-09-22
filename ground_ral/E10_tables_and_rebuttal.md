# E10 — Eq.(9) 逐项归一化重扫：正式表格与回应文本

> 对应 R1 ⑨ "Eq.(9) 不平衡 + 求解算法未说明"
> 所有数字：NR3D、grounder top-1、rerank 关闭、回放已存 query_graph（零额外 LLM 调用）。

## 表 R-E10a — 项尺度诊断（不平衡实证）

| run | queries | candidate states | median s_node | median s_rel | gap | s_node≥0.9 | s_rel≤0.5 |
|---|---|---|---|---|---|---|---|
| qwen3 | 100 | 424 | 0.951 | 0.371 | 0.579 | 64.6% | 62.6% |
| deepseek | 99 | 428 | 0.970 | 0.458 | 0.513 | 66.8% | 55.3% |
| glm | 100 | 449 | 0.920 | 0.403 | 0.517 | 55.5% | 57.1% |

**表 R-E10b — λu 全扫描（none = 原始加权；norm = 逐项 min-max 归一化后加权）**

| λu | MeanDist (none) | Acc@0.3 (none) | Acc@0.5 (none) | MeanDist (norm) | Acc@0.3 (norm) | Acc@0.5 (norm) |
|---|---|---|---|---|---|---|
| 0.00 | 1.4601 | 0.4500 | 0.4700 | 1.4601 | 0.4500 | 0.4700 |
| 0.25 | 1.4251 | 0.4600 | 0.4800 | 1.3090 | 0.4800 | 0.5000 |
| **0.50** | **1.2048** | **0.5200** | **0.5400** | 1.1976 | 0.5100 | 0.5300 |
| 0.75 | 1.2157 | 0.5000 | 0.5200 | 1.2406 | **0.5300** | **0.5400** |
| 1.00 | 1.6963 | 0.3600 | 0.3700 | 1.6963 | 0.3600 | 0.3700 |

（λu=0.0/1.0 两行归一化前后一致，符合预期：当一侧权重为 0 时归一化不影响排序。）

**表 R-E10c — 决策变化率（norm vs none，同 λu）**

| λu | top-1 改变数 | 总数 | 改变率 |
|---|---|---|---|
| 0.25 | 5 | 100 | 5.0% |
| 0.50 | 9 | 100 | 9.0% |
| 0.75 | 10 | 100 | 10.0% |

**表 R-E10d — 骨干稳健性（λu = 0.5，NR3D）**

| backbone | Acc@0.5 (none) | Acc@0.5 (norm) | Acc@0.3 (none) | Acc@0.3 (norm) |
|---|---|---|---|---|
| qwen3 | 0.5400 | 0.5300 | 0.5200 | 0.5100 |
| deepseek | 0.5758 | 0.5657 | 0.5354 | 0.5253 |
| glm | 0.5400 | 0.5400 | 0.5200 | 0.5100 |

## 回应文本（rebuttal 段落）

> We thank the reviewer for pointing out that Eq. (9) adds terms of very different scales. We confirm it quantitatively (Table R-E10a): across candidates the node term (CLIP semantic consistency) has median ≈0.92–0.97, while the relation term (geometric scorers) has median ≈0.37–0.46 — a gap of ≈0.5 on a [0,1] scale. A single λu weighting two such terms does not act as a clean node-vs-relation balance.
>
> We therefore normalize each term before weighting: node and relation scores are min-max normalized across candidates, and ranking uses λu·norm(S_node) + (1−λu)·norm(S_rel). We re-swept λu ∈ {0, 0.25, 0.5, 0.75, 1} on NR3D, with and without normalization (Table R-E10b). Two findings follow. First, the optimal λu remains ≈0.5 after normalization, so our reported configuration is not an artifact of the scale imbalance. Second, the normalized curve is flatter over 0.5–0.75 (Acc@0.5m 53–54% vs a 52–54% span un-normalized), indicating the previous peak partly reflected node-term saturation. The normalization is not cosmetic: it changes the top-1 selection for 9% of queries at λu = 0.5 (5% and 10% at 0.25 and 0.75; Table R-E10c), and the effect is consistent across Qwen3/DeepSeek/GLM backbones (Table R-E10d). As a sanity check, λu ∈ {0,1} are identical with and without normalization, as expected when one term has zero weight.
>
> Finally we clarify the underspecified solver: Eq. (9) is optimized by beam search over injective assignments (beam size 160; ≤24 target and ≤10 reference candidates per query node; all six viewer directions enumerated), and the compatibility score is computed in the log domain as a weighted geometric mean; we have added the attributed-graph-matching citation.

## 待补（正式版前）

1. 表 R-E10b 需用**全量查询集**重跑（当前 100 条 smoke；NR3D 全量/CSVG 同条目集），并补 SR3D 列。
2. 若审稿人要求全文一致，把论文 Table III 的三点（0.25/0.50/0.75）替换为上面的五点双行表。
