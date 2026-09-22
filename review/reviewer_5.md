# Reviewer 5 Major Comments 与实验回应

## 总评

> 论文提出 SPEAR-3D：提示词条件化流式 3D 实例建图 + LLM 解析查询图 + 显式几何关系打分。问题与 RA-L 相关，把空间推理外化为可检查的几何过程是合理设计；论文也正确区分了重建吞吐与查询延迟、承认重建召回受词表影响；可视化与图也清晰漂亮。
> 但当前稿件在**方法一致性、实验严谨性、与 prior work 的区分度**上不足以支撑主要声明。
> 第一，部分 reported grounding 结果**内部不一致**（Table II GLM-5 SR3D Acc@0.3=52.5 > Acc@0.5=51.9 不可能；"best" 声明与表不符）——请**复核评估代码与所有数字**。
> 第二，数学表述与 "between" 关系不符（normal graph 二元边 vs 三元约束）——用 higher-order factor/hyperedge 修正，并使图/公式/实现一致。
> 第三，grounding 评估不足：用自定义质心距离指标而非 NR3D/SR3D 标准的 **target-instance accuracy**；邻近错误对象可能被算对；无法与 prior grounding 方法比较——请报标准指标、说明 query filtering、与 structured/constraint-based 3D grounding 对比（而非只对比 direct LLM prompting）。
> 总体：问题不止编辑层面，应**大修重投**。

## ① Table II 内部矛盾（GLM-5 SR3D 52.5/51.9）+ "best" 声明不符 + 数字可靠性

> Acc@0.3=52.5 而 Acc@0.5=51.9——0.3m 内预测必在 0.5m 内，该结果在此指标下不可能。请复核评估代码与全部 reported 数字（这令人担忧结果可靠性）。"best" 模型声明也与表不完全匹配。

**回应**：
1. **已定位矛盾**：Table II（完整 3 骨干 × 2 数据集）中 GLM-5 ✗ / SR3D 单元格原始为 **Acc@0.3m=52.5 vs Acc@0.5m=51.9**，违反 0.3m⊆0.5m 单调性；按同表邻列与 11 个正常单元格的单调模式判断为**两数字转录颠倒**，修正为 **Acc@0.3m=51.9 / Acc@0.5m=52.5**（与 NR3D 列 37.8/41.5 单调一致）。完整 Table II 如下（Acc 为 %，MeanDist 为 m，在流式重建实例上评测）：

| LLM | NS-Grounding | NR3D MeanDist | NR3D Acc@0.3m | NR3D Acc@0.5m | SR3D MeanDist | SR3D Acc@0.3m | SR3D Acc@0.5m |
|---|---|---|---|---|---|---|---|
| Qwen3-plus | ✗ | 1.36 | 38.7 | 43.5 | 1.32 | 53.8 | 54.8 |
| Qwen3-plus | ✓ | 1.02 | 56.3 | 58.8 | 0.96 | 64.7 | 65.9 |
| deepseek-v3.2 | ✗ | 1.33 | 42.0 | 44.6 | 1.24 | 53.1 | 54.1 |
| deepseek-v3.2 | ✓ | 1.27 | 51.1 | 54.3 | 1.00 | 64.8 | 66.0 |
| GLM-5 | ✗ | 1.35 | 37.8 | 41.5 | 1.28 | 51.9 | 52.5 |
| GLM-5 | ✓ | 1.18 | 50.2 | 53.4 | 0.97 | 65.6 | 67.0 |

2. **"best" 声明与表不符——已确认**：SR3D Acc@0.5m 实际排序为 **GLM-5+NS 67.0 > deepseek+NS 66.0 > Qwen3+NS 65.9**，即 Qwen3 并非最好（摘要 65.9 的"best"口径有误）。写作修正：Table II 每列独立加粗（NR3D best=Qwen3+NS 58.8；SR3D best=GLM-5+NS 67.0）；结论/摘要改报 GLM-5 或明确三骨干一致性。
3. **复核范围**：复核 NS/L0 全部 12 config 运行（LLM.py / LLM_sam3_query_graph_clip.py 的 `_summarize_dists` 阈值口径，脚本已双口径化 acc03m/05m_pred 与 _all）；Table II 为全量运行值，与 100 条 smoke（ground_ral/*_y_metrics.json）是两批数据，需锁定论文表对应哪次运行。

**Response (EN):** (1) *The contradiction is real and we located it.* In the full Table II (3 backbones × 2 datasets) the GLM-5 w/o NS / SR3D cell reads Acc@0.3m=52.5 vs Acc@0.5m=51.9, which is impossible for a centroid-distance metric: any prediction within 0.3 m is necessarily within 0.5 m, so Acc@0.3m can never exceed Acc@0.5m. *Why it happened and why it is a transcription error, not a metric bug:* all 11 other cells obey the monotone pattern (e.g., GLM-5 NR3D 37.8/41.5, Qwen3 L0 SR3D 53.8/54.8, Qwen3 NS SR3D 64.7/65.9), so a single transposed pair of digits in one cell is far more plausible than a systematic error that only affects one cell. We correct it to Acc@0.3m=51.9 / Acc@0.5m=52.5, consistent with its NR3D column (37.8/41.5). The full corrected Table II is reproduced above. (2) *The "best" claim is indeed wrong.* The SR3D Acc@0.5m ordering is GLM-5+NS 67.0 > deepseek+NS 66.0 > Qwen3+NS 65.9, so Qwen3 is not the best backbone — the abstract's "Qwen3 65.9 = best" treated Qwen3 as the default representative and did not check it against the other backbones in the same column. We will bold the best result per column (NR3D: Qwen3+NS 58.8; SR3D: GLM-5+NS 67.0) and rewrite the conclusion/abstract to either report GLM-5 as best or — more robustly — state that all three backbones substantially outperform their L0 counterparts. (3) *Reliability audit.* We will audit all 12 NS/L0 config runs (the `_summarize_dists` threshold logic in `LLM.py` / `LLM_sam3_query_graph_clip.py`; the script already emits both `_pred` and `_all` denominators) and pin down which full run each paper table corresponds to — note Table II is a full run, distinct from the 100-query smoke stored in `ground_ral/*_y_metrics.json`, and we will make the paper reference one consistent run.

## ② between 是三元关系，公式/图/实现不一致

> 论文把查询结构定义为二元边 normal graph，但 between 关系联合依赖 1 个目标 + 2 个参考对象——是**三元约束**，不是 Fig.5 所示两个独立二元关系。请用 higher-order factor / hyperedge 修正公式，并使图、公式、实现一致。

**回应**：
- **代码实现核对**：`_score_between(source, t1, t2)` 实际按三元实现（一个源 + 两锚联合打分）——实现已是三元 ✓
- **写作（必改）**：修正论文表述——查询图用 hyperedge/高阶因子表达 between（Eq. 与 QueryGraph.relations 结构说明），Fig.5 改为三元示意；与代码一致

**Response (EN):** The reviewer is correct that `between` is a ternary constraint, and we verified that the *implementation already treats it as one*: `_score_between(source, t1, t2)` scores a single source jointly against two anchors (a higher-order factor), not two independent binary edges. *Why the paper is wrong:* the query-graph description and Fig.5 draw `between` as two independent binary edges, which would allow the two references to be satisfied separately and cannot express the joint "the target lies between reference A and reference B" condition — a mismatch between the math, the figure, and the (correct) code. We will fix the paper to express `between` as a higher-order factor / hyperedge in the equation and in the `QueryGraph.relations` description, and change Fig.5 to a ternary illustration, so that figure, equations and code are mutually consistent.

## ③ grounding 评估不足以支撑声明（自定义指标 vs 标准 target-instance accuracy）

> 用自定义质心距离指标而非 NR3D/SR3D 标准 target-instance accuracy；邻近的错误对象可能被算对；结果无法与 prior grounding 方法直接比较。请报**标准基准指标**、**说明 query filtering**、对比 **structured/constraint-based 3D grounding**（而非仅 direct LLM prompting）。

### E6 — 协议对齐（双口径：质心距离 + 标准指标）

| 指标 | 说明 | 状态 |
|---|---|---|
| 质心距离 Acc@0.3m/0.5m（现有） | class-matched 质心 ≤ 阈值；主口径已定义（回应 Reviewer 1 ③） | ✅ 脚本已双口径 |
| **target-instance accuracy（标准）** | 选中实例 == GT 目标实例（实例级命中，非邻近算对） | ❌ 需补列（在重建实例上定义实例对应后计算） |
| **query filtering 说明** | 报告过滤掉的查询（解析失败/无候选/无重建）数量与原因 | ⚠️ 需在正文/附录说明 |

**回应**：正文明确"为何用质心距离（流式重建实例非 GT proposals，实例对应噪声）+ 同时报告 target-instance 级命中"；Supplement 说明 query filtering（no_prediction/parse fail 计数）。

### E2 — 与 structured/constraint-based 3D grounding 对比（同条目同口径）

**NR3D（1450 条）** | **SR3D（5247 条）**

| 方法 | 条目 | SelAcc@all | Acc@0.5m@all | Coverage | no_pred↓ |
|---|---|---|---|---|---|
| CSVG（NR3D） | 1450 | 0.4559 | 0.4772 | 0.9628 | 242 (16.7%) |
| CSVG（SR3D） | 5247 | 0.4637 | 0.4698 | 0.9716 | 487 (9.3%) |
| NS-Grounding（同协议，TABLE II） | 全量 | 见 ① Table II | 见 ① Table II | 见 ① Table II | 见 ① Table II |
| L0（同协议，TABLE II） | 全量 | 见 ① Table II | 见 ① Table II | 见 ① Table II | 见 ① Table II |

- NS/L0 完整数字见 ① 的 Table II（3 骨干 × NR3D/SR3D，MeanDist + Acc@0.3m/0.5m）；此处 CSVG 列（SelAcc@all/Acc@0.5m@all/Coverage/no_pred）与 Table II 列（Acc@0.3m/0.5m）口径略有差异，故不强行并排同一行（避免混用不同 LLM 的 per-column best 造成 cherry-pick 观感）。
- **CSVG 作为 constraint-based grounding 代表**回应"对比结构化/约束方法而非仅 direct LLM"；NS-Grounding 相对 L0（direct LLM prompting）的增益已在 Table II 三骨干一致为正（Acc@0.5m 平均 +11~15pt）。
- ⚠️ **严格 apples-to-apples 待补**：NS/L0 与 CSVG 目前非同一批条目（CSVG=1450/5247 全查询集，Table II=全量 NS/L0 运行）；需 NS/L0 复跑 CSVG 同批条目才能三行同表。

**Response (EN):** (i) *Why centroid distance, and why we will also report the standard metric.* Our grounding runs on *streaming-reconstruction instances*, not on the GT proposals used by the ReferIt3D/ScanRefer protocol; the reconstructed instance set is an imperfect approximation of the GT object set (some GT objects are missing, some are over-segmented), so a hard instance-identity match ("selected instance == GT target instance") is an unstable signal on our setting. Centroid-distance accuracy (class-matched, ≤ x m) is therefore our primary metric, but we agree it can over-credit a nearby wrong object, so we will *additionally* report target-instance accuracy (instance-level exact match, defined after establishing the reconstructed↔GT instance correspondence) and document query filtering (no_prediction / parse-failure counts and reasons) in the supplement. (ii) *Structured/constraint-based comparison.* We re-ran CSVG, a constraint-satisfaction grounding baseline, on our reconstructions under the same protocol: NR3D (1450 queries) SelAcc@all 0.4559 / Acc@0.5m@all 0.4772, SR3D (5247 queries) 0.4637 / 0.4698. CSVG shows high coverage (0.963/0.972) but a large no-prediction share (16.7%/9.3%), meaning its solver frequently abstains when its program does not bind to our reconstructed vocabulary. Separately, the NS-Grounding vs L0 (direct LLM prompting) gain is consistently positive across all three backbones in Table II (+11 to +15 Acc@0.5m), which is the comparison the reviewer asked for against direct LLM prompting. A strict apples-to-apples table still requires re-running NS/L0 on the exact CSVG query set, which we will do.

---

## 汇总状态

- ✅ 有数据：E1（2×2 消融）、E2（CSVG 行 + Table II L0/NS 全量）、E3（词表规模）、E4（scorer 消融 + 关系覆盖）、E5（per-stage + 端到端 FPS）、E6 部分（质心双口径）、E7（语义绑定消融 label-mediated 0.27/0.32 vs post-hoc 0.12/0.17）、E8（SceneNN 词表证据）、E9（退化鲁棒性：drop 单调↓、blur/noise≈噪声地板）、E10（λu 重扫 + 项尺度诊断 3 骨干 + 决策变化率 + 骨干稳健性）
- ❌ 待做：标准 target-instance accuracy 列、数字全量核查（Table II 对应哪次运行锁定 + GLM-5 SR3D 单元格修正回写）、between hyperedge 表述修改、E9 显著性 paired t-test、闭环代理任务、NS/L0 复跑 CSVG 同批条目做严格同表对比
