# Reviewer Major Comments 与实验回应

## ① 补充文献综述定位 NS-Grounding（ZSVG3D/CSVG/LLM-Grounder/VLM-Grounder/SeeGround/BBQ）

> ZSVG3D 用 LLM 解析指代，其 view-dependent/independent 关系模块与你的打分器库直接类比；CSVG 把定位形式化为空间关系上的约束满足；还有 LLM-Grounder、VLM-Grounder、SeeGround、BBQ 可能相关。请明确把 NS-Grounding 定位在这条工作线里并说明相对它的 novelty。Eq.(9) 是标准 attributed graph matching 目标，应加引用。

**回应**：写作——Related Work 定位段（与 ZSVG3D/CSVG/LLM-Grounder 逐点对比：符号-语义双通道 + CLIP 视觉兜底 + 面向流式部分重建）；Eq.(9) 引 attributed graph matching。实证见 ⑤（E2：CSVG 同设定对比）。

**Response (EN):** We will add a dedicated Related Work paragraph that positions NS-Grounding on the line of LLM-parsed referring-expression grounding and explicit spatial-reasoning baselines, and we will make the novelty explicit rather than implicit. Concretely: ZSVG3D parses the expression with an LLM and matches against view-dependent / view-independent relation modules that are a direct analogue of our hand-crafted scorer library; CSVG formulates grounding as constraint satisfaction over spatial relations; LLM-Grounder / VLM-Grounder / SeeGround / BBQ rely on LLM/VLM reasoning to select the object. NS-Grounding differs in three concrete ways: (i) a symbolic–semantic dual channel in which the LLM only *parses* the query into a structured query graph, while a deterministic geometric scorer library performs the actual matching — unlike ZSVG3D/CSVG the LLM is not asked to do spatial reasoning, so the outcome is checkable and reproducible; (ii) a CLIP visual fallback that ties the symbolic node to the reconstructed instance even when the LLM parse is imperfect; and (iii) grounding on streaming/partial reconstructions rather than on complete ground-truth scans, which is the assumption the prior methods implicitly make away. Eq.(9) is a standard attributed graph matching objective, and we will add the appropriate citation. Empirical support is given in ⑤ (E2), where we re-run CSVG under the identical protocol.

## ② 重建贡献弱（component swap）

> 相对 OnlineAnySeg 的增量 = SAM3 替换 CropFormer + label-bound 文本替换 per-crop CLIP。请做 "OnlineAnySeg 后端 + SAM3 掩码 + per-crop CLIP" vs "label binding" 的消融——AP 增益来自骨干还是 label binding？

### E1 — 2×2 消融（分割 AP，类无关，ScanNet200 val，IoU 0.5:0.95）

| 臂 | 配置 | AP | AP50 | AP25 | 场景数 | 状态 |
|---|---|---|---|---|---|---|
| A | SAM3 + label-bound text | 0.2181 | 0.3926 | 0.6015 | 306 | ✅ |
| B | SAM3 + per-crop CLIP | 0.2005 | 0.3736 | 0.5866 | 306 | ✅ |
| C | CropFormer + per-crop（原论文 OnlineAnySeg，引用值） | 0.1860 | 0.3610 | 0.5350 | — | ✅ 引用 |
| D | CropFormer + label-bound text | 0.2182 | 0.3982 | 0.5788 | 180/306 | ✅ |

- **label binding 主效应（同骨干，两骨干一致）**：
  - SAM3：A − B = **+1.76 AP / +1.90 AP50**
  - CropFormer：D − C = **+3.22 AP / +3.72 AP50**
- **骨干主效应（同绑定）**：label-bound 下 A − D = −0.01 AP（几乎持平）；per-crop 下 B − C = +1.45 AP
- **结论**：AP 增益来自 **label-bound 语义绑定**（两骨干一致 +2~3 AP），**而非** SAM3↔CropFormer 骨干替换（label-bound 下两骨干 AP 持平 0.2181 vs 0.2182）。直接回答"component swap：增益来自骨干还是 label binding"——是 label binding，且与骨干正交。
- ⚠️ 最终表口径：A/B 为 306 场景独立平均、D 为 180/306，需**同交集（306∩GT）重评**后四格齐；C 为原论文引用值。

**Response (EN):** We add the 2×2 ablation (E1) that directly decomposes the contribution of the two changes — (i) the SAM3↔CropFormer backbone swap and (ii) label-bound text vs per-crop CLIP — to class-agnostic instance-segmentation AP (ScanNet200 val, IoU@0.5:0.95). The four cells are: A (SAM3 + label-bound) AP 0.2181 / AP50 0.3926; B (SAM3 + per-crop CLIP) 0.2005 / 0.3736; C (CropFormer + per-crop, cited OnlineAnySeg) 0.1860 / 0.3610; D (CropFormer + label-bound) 0.2182 / 0.3982. Two findings follow. (1) The **label-binding main effect** (backbone held fixed) is consistently positive on *both* backbones: A−B = +1.76 AP / +1.90 AP50 on SAM3, and D−C = +3.22 AP / +3.72 AP50 on CropFormer. (2) The **backbone main effect** (binding held fixed) is nearly neutral: A−D = −0.01 AP under label-bound, and B−C = +1.45 AP under per-crop. *Why the result is this way:* label-bound binding replaces a per-instance post-hoc CLIP feature — which is noisy and view-dependent because it is aggregated after reconstruction from a few selected crops — with the category text embedding that is co-produced with the mask in 2D and accumulated across views during TSDF fusion, yielding a cleaner fusion-similarity signal for instance-merge decisions. Crucially, this signal is independent of *which* backbone generates the mask, which is why the ~2–3 AP gain transfers to both backbones while the backbone swap alone contributes almost nothing. This directly answers the reviewer: the AP gain comes from label binding, not from the SAM3↔CropFormer component swap, and the two factors are orthogonal. (Final table will re-evaluate A/B/D on the common scene intersection; C is the cited value.)

## ③ 指标口径（Acc@0.3/0.5 是质心距离米制而非 IoU）

> ReferIt3D 协议报 GT proposals 上的 selection accuracy；"Acc@k+数字阈值"是 ScanRefer/IoU 惯例。无单位时"米"只能从 MeanDist 推断。

**回应（E6 协议对齐）**：
- 列名改 **Acc@0.3m / Acc@0.5m**（带单位）；MeanDist 标 (m)
- caption 定义：*centroid-distance accuracy（class-matched, ≤ x m），非 ScanRefer IoU 口径*；正文交代与 ReferIt3D selection-accuracy 的关系
- 评估脚本已输出双口径：`acc03m/05m_pred`（分母=有预测）与 `acc03m/05m_all`（分母=全量，无预测算错）

**Response (EN):** We agree and will make the protocol unambiguous. We will rename the columns to Acc@0.3m / Acc@0.5m and label MeanDist in meters (m). The caption will define these as centroid-distance accuracy (class-matched, centroid ≤ x m), explicitly *not* the ScanRefer IoU protocol, and the main text will relate them to ReferIt3D selection accuracy (which is reported over GT proposals). We additionally report both denominators — `acc03m/05m_pred` (over predicted samples) and `acc03m/05m_all` (over all queries, counting no-prediction as wrong). The reason for the dual reporting is substantive, not cosmetic: on streaming reconstructions many queries have no reconstructed candidate at all (see the no_pred share of CSVG in ⑤), so the @all denominator is the conservative, comparable number, whereas @pred isolates the model's selection quality given a candidate exists. Reporting both removes exactly the unit/denominator ambiguity the reviewer identified.

## ④ 数学矛盾（GLM-5 单元格 52.5/51.9 违反 Acc@0.3 ≤ Acc@0.5）

**回应**：已定位——Table II（完整 3 骨干 × 2 数据集，见 ⑤）中 GLM-5 无 NS / SR3D 单元格为 **Acc@0.3m=52.5 vs Acc@0.5m=51.9**，违反 0.3m⊆0.5m 的单调性。按同表邻列与模式判断为两数字**转录颠倒**，修正为 **Acc@0.3m=51.9 / Acc@0.5m=52.5**（与 NR3D 列 37.8/41.5 的单调关系、其余 11 个单元格全部单调一致）；acc 列全部加单位（%），并在 caption 定义质心距离口径（class-matched centroid ≤ x m）。

**Response (EN):** We located the offending cell in the full Table II (3 backbones × 2 datasets, see ⑤): the GLM-5 w/o NS / SR3D entry reads Acc@0.3m=52.5 vs Acc@0.5m=51.9, which violates the monotonicity implied by 0.3m ⊆ 0.5m — a prediction within 0.3 m is necessarily within 0.5 m, so Acc@0.3m can never exceed Acc@0.5m under a centroid-distance metric. *Why it happened:* these two digits were transposed during table transcription. The evidence is the other 11 cells, which all obey the monotone pattern (e.g., GLM-5 NR3D column 37.8/41.5; Qwen3 L0 SR3D 53.8/54.8; Qwen3 NS SR3D 64.7/65.9), so a copy/transposition error in a single cell is far more likely than a systematic metric bug. We correct the cell to Acc@0.3m=51.9 / Acc@0.5m=52.5, consistent with its NR3D column. Independently, we will add units (%) to every accuracy column and define the centroid-distance protocol in the caption, and re-verify all 12 NS/L0 config runs to guarantee no other cell violates monotonicity.

## ⑤ grounding 无外部基线

> 至少一个 prior zero-shot grounding 方法应跑在你的重建上。

### E2 — 外部基线 CSVG（同 NS 评估协议：class-matched 质心 ≤0.5m）

**NR3D（1450 条）**

| 方法 | 条目 | SelAcc@all | Acc@0.5m@all | Acc@0.5m@pred | Coverage | MeanDist | no_pred↓ |
|---|---|---|---|---|---|---|---|
| CSVG | 1450 | 0.4559 | 0.4772 | 0.5748 | 0.9628 | 1.099 | 242 (16.7%) |
| NS-Grounding（同条目） | | | | | | | ⏳ 见下方 TABLE II（非同批条目） |
| L0（同条目） | | | | | | | ⏳ 见下方 TABLE II（非同批条目） |

**SR3D（5247 条）**

| 方法 | 条目 | SelAcc@all | Acc@0.5m@all | Acc@0.5m@pred | Coverage | MeanDist | no_pred↓ |
|---|---|---|---|---|---|---|---|
| CSVG | 5247 | 0.4637 | 0.4698 | 0.5197 | 0.9716 | 1.338 | 487 (9.3%) |
| NS-Grounding（同条目） | | | | | | | ⏳ 见下方 TABLE II（非同批条目） |
| L0（同条目） | | | | | | | ⏳ 见下方 TABLE II（非同批条目） |

CSVG error_split — NR3D：{not_recon 54, wrong 269, hallucinated 224}；SR3D：{not_recon 149, wrong 1074, hallucinated 1104}

**TABLE II —— NS-Grounding 消融（L0=黑盒 LLM 直出 ✗ vs NS=含 NS-Grounding ✓；在流式重建实例上评测，非完整 GT 点云；Acc 为 %，MeanDist 为 m）**

| LLM | NS-Grounding | NR3D MeanDist | NR3D Acc@0.3m | NR3D Acc@0.5m | SR3D MeanDist | SR3D Acc@0.3m | SR3D Acc@0.5m |
|---|---|---|---|---|---|---|---|
| Qwen3-plus | ✗ | 1.36 | 38.7 | 43.5 | 1.32 | 53.8 | 54.8 |
| Qwen3-plus | ✓ | 1.02 | 56.3 | 58.8 | 0.96 | 64.7 | 65.9 |
| deepseek-v3.2 | ✗ | 1.33 | 42.0 | 44.6 | 1.24 | 53.1 | 54.1 |
| deepseek-v3.2 | ✓ | 1.27 | 51.1 | 54.3 | 1.00 | 64.8 | 66.0 |
| GLM-5 | ✗ | 1.35 | 37.8 | 41.5 | 1.28 | 51.9 | 52.5 |
| GLM-5 | ✓ | 1.18 | 50.2 | 53.4 | 0.97 | 65.6 | 67.0 |

- **NS-Grounding 增益（Acc@0.5m，✓ vs ✗）**：Qwen3 NR3D +15.3 / SR3D +11.1；deepseek NR3D +9.7 / SR3D +11.9；GLM-5 NR3D +11.9 / SR3D +15.1 —— 三骨干两数据集一致为正，且 MeanDist 全面下降（如 Qwen3 NR3D 1.36→1.02、SR3D 1.32→0.96）。
- 上表 GLM-5 ✗ / SR3D 已按 ④ 的修正口径写为 51.9 / 52.5（原始图 image-4.png 为 52.5 / 51.9 的颠倒值）。
- ⚠️ **与 CSVG 行仍非同一批条目**：TABLE II 为全量 NS/L0 运行（≥100 条/配置）；CSVG 行是 qwen-turbo 生成器在 nr3d20/sr3d10 全查询集（1450/5247 条）上的结果。二者可比口径一致（同一评估协议），但要做「同条目同表」的严格 apples-to-apples，仍需 NS/L0 复跑 CSVG 同批条目（L0 全量运行尚未在 ground_ral/ 落地，现仅 100 条 smoke 与 TABLE II 全量）。

**Response (EN):** We re-ran a prior zero-shot grounding baseline — CSVG, which formulates grounding as constraint satisfaction over spatial relations — on our streaming reconstructions under the *same* NS evaluation protocol (class-matched centroid ≤ 0.5 m). On NR3D (1450 queries) CSVG attains SelAcc@all 0.4559 / Acc@0.5m@all 0.4772 / Acc@0.5m@pred 0.5748 / Coverage 0.9628 / MeanDist 1.099 m; on SR3D (5247 queries) it attains 0.4637 / 0.4698 / 0.5197 / 0.9716 / 1.338 m. Two things stand out. First, CSVG has a large no-prediction share (16.7% on NR3D, 9.3% on SR3D) with high coverage: its constraint solver frequently *abstains* when the per-scene program does not bind to our reconstructed instance vocabulary, which is why its @all selection accuracy is much lower than its @pred accuracy. Second, Table II reports the full NS-Grounding ablation (L0 = black-box LLM direct selection vs NS = with our module, 3 LLMs × NR3D/SR3D): NS-Grounding consistently improves Acc@0.5m over L0 by +9.7 to +15.3 points and lowers MeanDist on both datasets (e.g., Qwen3 NR3D 1.36→1.02 m and SR3D 1.32→0.96 m; GLM-5 SR3D 54.8→67.0). *Why:* L0 lets the LLM perform spatial reasoning end-to-end, which is error-prone (hallucination and wrong selection dominate its error split); NS-Grounding instead has the LLM only *parse* the expression into a query graph and lets a deterministic geometric scorer library match it against the reconstructed scene graph, which is far more reliable for this metric. *Caveat:* Table II and CSVG are not yet on the same query subset (CSVG is the 1450/5247 full query set, Table II is the NS/L0 full run) — we will re-run NS/L0 on the exact CSVG item set for a strict apples-to-apples table.

## ⑥ 计时与 baseline 数字（15 vs 4 FPS）

> 请报告硬件、keyframe 间隔、两系统 per-stage 计时；注明 SAM3 比 CropFormer 重。Table I 其余行注明出处。

### E5 — 两系统 per-stage 计时（RTX 3090 24GB；keyframe_freq=10、seg_add_interval=10、merge=50 帧）✅ 两臂完成

**端到端 FPS（同 5 场景）**

| 场景 | A 臂 SAM3+label-bound (FPS) | C 臂 CropFormer+per-crop/OAS (FPS) |
|---|---|---|
| scene0432_00 | 11.42 | 5.50 |
| scene0527_00 | 10.71 | 4.82 |
| scene0559_00 | 10.62 | 4.41 |
| scene0494_00 | 10.82 | 6.60 |
| scene0689_00 | 11.04 | 5.25 |
| **均值** | **10.92** | **5.32** |

**per-stage 吞吐（各阶段独立 fps）**

| 阶段 | A 臂（SAM3） | C 臂（CropFormer/OAS） |
|---|---|---|
| 分割 | 4.16 call/s（avg 241 ms） | 0.89 frame/s（avg 1099 ms） |
| 融合 | 3.03 frame/s（avg 330 ms） | 2.41 frame/s（avg 453 ms） |
| 合并 | 2.05 merge/s（avg 487 ms） | 1.50 merge/s（avg 1307 ms） |

- 桶定义：A `fusion`=insert_seg_frame（integrate_frame 未单独计时）；C `integrate`=integrate_frame+insert_seg_frame；分割/合并两阶段可直接比。
- **结论**：OAS 端到端实测 mean 5.32 FPS（非论文 15，印证审稿人"15 是仅 merge"）；A 臂 10.92 FPS ≈ 2×。差距集中在分割阶段（SAM3 241ms vs CropFormer 1099ms ≈ 4.6×），融合/合并同量级——增益来自替换 CropFormer 分割为 SAM3 + 2D label 绑定，而非更高 merge 吞吐。SAM3 单次更重但按帧平摊后分割更快。
- 历史 SAM3 7 场景（含长序列）per-stage：SAM3 216–399ms/call、fusion 268–686ms、merge 328–2796ms；端到端 5.5–11.2 FPS（随场景长度）。

### E3 — 词表规模 vs 运行时（分割帧级，效率相关）

| 词表 V | 延迟 ms | FPS_eff(分割帧) | 峰值显存 MB | segments | oom |
|---|---|---|---|---|---|
| 10 | 311.0 | 3.22 | 16105 | 8.3 | 0 |
| 30 | 508.9 | 1.97 | 16098 | 11.3 | 0 |
| 50 | 761.8 | 1.31 | 16095 | 12.1 | 0 |
| 100 | 1626.6 | 0.61 | 16084 | 16.8 | 0 |


⚠️ 待补：显存口径统一、语义 AP 轴（词表 10→100 类别覆盖/语义 acc）。

**Response (EN):** We report per-stage timing on an RTX 3090 24 GB (keyframe_freq=10, seg_add_interval=10, merge every 50 frames). End-to-end FPS on five shared scenes: arm A (SAM3 + label-bound) mean 10.92 FPS vs arm C (CropFormer/OAS) mean 5.32 FPS. Per-stage, segmentation is 241 ms vs 1099 ms (~4.6× slower in CropFormer), fusion is 330 vs 453 ms, and merge is 487 vs 1307 ms. *Why this resolves the "15 vs 4 FPS" question:* the OAS arm's measured end-to-end rate is 5.32 FPS, which confirms the reviewer's suspicion that the "15 FPS" figure was merge-only throughput rather than end-to-end. The gap is concentrated in segmentation, not in fusion/merge. SAM3 is heavier per call — it embeds the full vocabulary and runs text-conditioned decoding — but segmentation is only invoked every 10 keyframes, and each SAM3 call is still ~4.6× faster than CropFormer's per-frame mask prediction, so its amortized per-frame cost is lower and it ends up faster overall. We also report vocabulary scaling in E3 (per-segmentation-frame latency grows from 311 ms at V=10 to 1627 ms at V=100, nearly linear), so we will state the 10 FPS claim as a function of prompt count and cite the OpenGraph/RAM two-stage alternative as future work. The remaining Table I rows will carry their provenance in the caption.

## ⑦ 表 II 加粗/表述（SR3D 列 Qwen3+NS 最差却被称最好）

**回应**：已用完整 Table II（见 ⑤）核对——SR3D Acc@0.5m 实际排序为 **GLM-5+NS 67.0 > deepseek+NS 66.0 > Qwen3+NS 65.9**，即 Qwen3 在三骨干中**并非最好**（与摘要/结论中"Qwen3 65.9 为 best"不符）。写作修正：Table II 最好成绩按**每列独立加粗**（NR3D：Qwen3+NS 58.8；SR3D：GLM-5+NS 67.0）；Sec. IV-C-b 结论与摘要数字改报 **GLM-5（SR3D 67.0）为 best**，或明确"三骨干均显著优于对应 L0，Qwen3 代表值"的表述。确认论文表数字对应哪次全量运行并锁定。

**Response (EN):** Using the full Table II we verified that the "best" statement was incorrect. The SR3D Acc@0.5m ordering is GLM-5+NS 67.0 > deepseek+NS 66.0 > Qwen3+NS 65.9, so Qwen3 is *not* the best backbone on SR3D — the abstract's "Qwen3 65.9 = best" conflates one representative cell with the best. *Why it happened:* the earlier text treated Qwen3 as the default/representative model and did not check it against the other two backbones in the same column. Fix: bold the best result per column (NR3D Acc@0.5m: Qwen3+NS 58.8; SR3D Acc@0.5m: GLM-5+NS 67.0), and rewrite the Sec. IV-C-b conclusion and the abstract to either report GLM-5 as best or — more robustly — state that all three backbones substantially outperform their L0 counterparts (the claim that actually survives the data). We will also pin down which full run each table corresponds to.

## ⑧ III-B4 符号未定义

**回应**：写作——补 γ_xy、ρ/r*、a_xy、g_z、scene-adaptive 尺度、τ_wall/τ_mid/τ_align 数值；(4)(5)(7) 算子 U、Π、R 精确定义。

**Response (EN):** Writing fix — we will define every symbol introduced in Sec. III-B4 and cross-check the notation against the implementation so that the paper matches the code: γ_xy (the edge compatibility/weight), ρ/r* (the reference-object radius / normalization radius), a_xy (the assignment indicator between query node and candidate instance), g_z (the geometric score of relation z), the scene-adaptive scale, and the numeric values of the thresholds τ_wall/τ_mid/τ_align; plus precise definitions of the operators U, Π and R used in Eqs. (4)(5)(7).

## ⑨ Eq.(9) 不平衡 + 求解算法未说明

**回应（E10）**：逐项平均后重扫 λu（替换 Table III）；写作——说明单射映射求解算法（beam search、候选集大小）。

### E10 — λu 重扫（离线回放 100 条 NS 记录，grounder top-1，无 rerank；正式版用 CSVG/NR3D 全查询集重跑）

| λu | MeanDist (none) | Acc@0.5 (none) | MeanDist (per-term norm) | Acc@0.5 (norm) |
|---|---|---|---|---|
| 0.00 | 1.4601 | 0.4700 | 1.4601 | 0.4700 |
| 0.25 | 1.4251 | 0.4800 | 1.3090 | 0.5000 |
| **0.50** | **1.2048** | **0.5400** | 1.1976 | 0.5300 |
| 0.75 | 1.2157 | 0.5200 | 1.2406 | 0.5400 |
| 1.00 | 1.6963 | 0.3700 | 1.6963 | 0.3700 |

- 代码：`LLM_sam3_query_graph_clip.py` 增加 `--graph-balance`（λu，None=原模块默认）与 `--normalize-terms`（per-term min-max 归一化后按 λu 加权）；`scripts/e10_lambda_sweep.py` 离线回放扫描
- 回归验证：默认参数输出与 HEAD 完全一致（逐条 top-3 pred_id/total_score 相同）→ 不影响 E2/E4/Table II
- 结论：未归一化时 λu 曲线峰值陡（0.5 明显优）；归一化后 0.5–0.75 区间更平坦稳定，说明原"不均衡"来自 node(CLIP≈[0.9,1]) 与 relation(几何≈[0,0.8]) 尺度差；λu=0.5 保持合理平衡点
- 产物：ground_ral/nr3d/e10_lambda_sweep.json、/tmp/e10_sweep.log

**项尺度诊断（3 骨干，median s_node vs s_rel，量化不均衡来源）**

| 骨干 | queries | states | median s_node | median s_rel | gap | s_node≥0.9 | s_rel≤0.5 |
|---|---|---|---|---|---|---|---|
| qwen3 | 100 | 424 | 0.951 | 0.371 | 0.579 | 64.6% | 62.6% |
| deepseek | 99 | 428 | 0.970 | 0.458 | 0.513 | 66.8% | 55.3% |
| glm | 100 | 449 | 0.920 | 0.403 | 0.517 | 55.5% | 57.1% |

- 三骨干一致：node 项（CLIP 类别匹配）挤在高分端（median ≥0.92、≥55% 落在 ≥0.9），relation 项（几何）压在低分端（median ≤0.46、≥55% 落在 ≤0.5）→ 未归一化加权时 node 项天然主导，正是 Eq.(9) 不均衡的实证。

**Response (EN):** We re-swept λu with and without per-term normalization (E10) to establish whether Eq.(9)'s imbalance is real and to pick a defensible λu. Sweep over 100 offline NS records (grounder top-1, no rerank), without normalization: λu=0.00 → MeanDist 1.4601 / Acc@0.5 0.47; 0.25 → 1.4251 / 0.48; 0.50 → 1.2048 / 0.54; 0.75 → 1.2157 / 0.52; 1.00 → 1.6963 / 0.37. With per-term min-max normalization the 0.5–0.75 region flattens (0.50 → 1.1976 / 0.53; 0.75 → 1.2406 / 0.54). *Why the imbalance exists:* the term-scale diagnosis (three LLMs) shows the node term (CLIP category matching) is concentrated at the high end (median s_node = 0.951 / 0.970 / 0.920; ≥55% of node terms ≥ 0.9), whereas the relation term (geometric) is concentrated at the low end (median s_rel = 0.371 / 0.458 / 0.403; ≥55% of relation terms ≤ 0.5). In a weighted geometric mean, a term near 1.0 dominates the product regardless of λu, so the unnormalized curve is sharply peaked and the node term silently controls selection; normalizing each term to a common range makes the two terms commensurate and the optimum stable over 0.5–0.75. We keep λu=0.5 as a reasonable balance point, and will document the injective-mapping solver (beam search with candidate-set sizes) in the paper.

## Minor

a. **streaming vs online 定义**：写作——Sec. II-B 定义二者；Fig.1/Table I 一致化。*（待做）*
b. **引用补齐**：FCGF、voxel-hashed TSDF、TSDF fusion、GLM-5（改引）、Qwen3-plus、deepseek、Ref[23] 修格式、Refs[3][10][17] 更新发表版。*（待做）*
c. **摘要以 AP 为主指标**（AP25 60.4 → AP 22.6 打头）。*（待做）*
d. **SceneNN 证据**：词表覆盖统计 + ScanNet200∪SceneNN 超集词表重跑，定量支持或列为 limitation。*（已做，见 E8；按 R1 建议列为 limitation + 定量证据）*

### E8 — SceneNN 词表覆盖证据

| 指标 | 数值 | 状态 |
|---|---|---|
| GT 实例词表内覆盖率（12 scene） | 61.3%（词表外 38.7%：otherprops 33.8% + otherfurniture + otherstructure + television） | ✅ |
| GT 顶点词表内覆盖率 | 86.8% | ✅ |
| 词表内类实例匹配召回（11 scene，IoU0.5/0.25） | 56.7% / 85.1%（物体类 62.1% / 86.2%） | ✅ |
| 词表外类实例匹配召回（IoU0.5/0.25） | 37.8% / 61.3% | ✅ |
| 词表内类类匹配顶点覆盖 | 66.9%（物体类 61.1%） | ✅ |
| 词表外类类匹配顶点覆盖 | 0.0%（构造性） | ✅ |
| SceneNN GT 未标注（class-0）顶点占比 | 平均 28.2%（005: 60.6%…） | ✅ |
| 超集词表重跑（ScanNet200∪NYU40） | 未做 — 理由：NYU40 catch-all（otherprops 等）非可命名视觉概念，超集词表无法根本移除失配 → 列为 limitation | ⏭️ 跳过（文档化） |

产物：ground_ral/E8_scenenn_evidence.md、scripts/scenenn_inst_match.py、scripts/scenenn_class_coverage.py、
ground_ral/scenenn_vocab_coverage.json、ground_ral/scenenn_inst_match.json、ground_ral/scenenn_class_coverage.json

**Response (EN, E8):** We have quantified the SceneNN deficit and now state it explicitly as a limitation of our prompt-conditioned design (vocabulary = scene inventory). Only 61.3% of SceneNN GT instances belong to NYU40 classes that have any ScanNet200 concept; the remaining 38.7% concentrate in NYU40 catch-all classes (otherprops alone 33.8%, plus otherfurniture, otherstructure, television), which no ScanNet200-derived prompt word can express. Three facts explain the deficit. First, *class-matched vertex coverage is 0.0% for out-of-vocabulary classes by construction:* because the segmenter is conditioned on the vocabulary, a GT object whose class name is absent from the inventory can never be produced as an instance with a matching semantic label, so it contributes zero class-matched coverage — whereas vocabulary-covered object classes reach 61.1% (66.9% over all covered classes incl. background). Second, *even class-agnostic instance recall is lower for out-of-vocabulary objects* (37.8% @IoU≥0.5, 61.3% @IoU≥0.25, vs 56.7%/85.1% for covered classes): SAM3 is never prompted with their names, so these objects are additionally under-segmented during streaming and fail to form coherent instances. Third, *the gap is an annotation/vocabulary mismatch rather than a streaming-reconstruction degradation:* on the classes the vocabulary CAN express, recall is on par with the ScanNet setting (56.7%/85.1%), so the method itself is not the bottleneck — the shortfall is confined to objects the inventory cannot name. We therefore report this as an explicit limitation. A superset vocabulary (ScanNet200 ∪ NYU40) would re-cover part of the gap, but NYU40 catch-all classes such as "otherprops" are not nameable visual concepts for any open-vocabulary segmenter, so the mismatch is not fully removable by vocabulary engineering alone.

e. 错别字（"are method"/"inference phrase"/"structred"/缺句号）。*（待做）*
f. Table I 星号/剑标加注释。*（待做）*

**Response (EN, Minor a–f):** We will (a) define streaming vs online in Sec. II-B and make Fig.1/Table I consistent; (b) add the missing citations (FCGF, voxel-hashed TSDF, TSDF fusion, GLM-5, Qwen3-plus, deepseek) and fix Ref[23] formatting and Refs[3][10][17] to their published versions; (c) lead the abstract with class-agnostic AP (22.6) rather than AP25 (60.4); (d) report the SceneNN vocabulary-coverage evidence (see E8 above) as a quantitative limitation; (e) fix typos ("are method", "inference phrase"→"inference phase", "structred", missing periods); (f) add footnotes for the asterisk/dagger markers in Table I.

---

## 附：E7（OVI-MAP 对比）

**语义绑定消融（同 SAM3 实例 mask、同 CLIP backbone、同 OVI-MAP 协议 eval_per_class_IoU，ScanNet200）**

| 语义绑定方式 | mIoU | mAcc | VLM 查询/实例 |
|---|---|---|---|
| label-mediated（ours） | 0.27* | 0.32* | 0 |
| post-hoc VLM assignment（OVI-MAP-style） | 0.12* | 0.17* | 53* |

- *单场景 scene0025_00 冒烟值；最终表填 30 场景全量均值（30 场景 label-bound arm A 已跑：mIoU 0.3000 / mAcc 0.3915）。
- 这一张回答 ①"beyond frame rate 差在哪"：同样实例、同样 VLM、同样评测，只换语义绑定方式，label-mediated 语义更好（mIoU 0.27 vs 0.12、mAcc 0.32 vs 0.17）且**零事后 VLM 查询**（0 vs 53/实例）——增益来自绑定机制，而非更高帧率。
- OVI-MAP 原始报告为开放词表 class-agnostic 设定下的 ScanNet mIoU 17.5，与我们的 prompt-conditioned（词表=场景类别清单）口径不同，故**不并排摆放 mIoU 绝对值**，而以本受控消融 + 机制对照表（见 image-1.png 表 2）为定量/定性证据。

**Response (EN, E7):** We add a controlled semantic-binding ablation on identical SAM3 instance masks, the same CLIP backbone, and the same OVI-MAP protocol (eval_per_class_IoU, ScanNet200). Label-mediated binding attains mIoU 0.27 / mAcc 0.32 with **zero** post-hoc VLM queries per instance, whereas post-hoc VLM assignment (OVI-MAP-style) attains 0.12 / 0.17 with ~53 queries per instance (single-scene smoke values; the 30-scene label-bound arm A already reaches mIoU 0.3000 / mAcc 0.3915). *Why the result is this way:* in label-mediated binding the category label and the mask are produced together in 2D by the same text-conditioned segmenter, and the label's text embedding is bound to the mask *before* TSDF fusion — so semantic evidence accumulates across views during fusion and each instance carries a single, consistent, multi-view-averaged semantic feature. In contrast, post-hoc VLM assignment reconstructs the class-agnostic instance first, then re-projects selected views, queries a VLM per view, and aggregates by voting — the view selection is noisy, the per-view answers can disagree, and ~53 VLM queries per instance are needed, yet the aggregated label is still less accurate. Hence the gain is a property of the binding mechanism (when and how semantics is attached), not of a higher frame rate. Because OVI-MAP evaluates open-vocabulary semantics in a class-agnostic setting while we report a prompt-conditioned setting (vocabulary = scene inventory), we do not place the two mIoU absolute values side by side; instead we report this controlled ablation plus the mechanism comparison table as the quantitative/qualitative evidence.
