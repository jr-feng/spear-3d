# Reviewer 3 Major Comments 与实验回应

## Summary（审稿人总结）

> 论文提出 SPEAR-3D 两阶段框架：(1) 流式 3D 语义实例重建（RTX 3090 上 10 FPS，用推理期词表条件化 SAM3 分割与 CLIP 特征绑定，复用 OnlineAnySeg 重建后端）；(2) NS-Grounding：LLM 把自然语言查询解析为结构化查询图，用一组手工空间关系打分器与重建实例导出的场景图匹配。
> 方向合理、动机（用显式几何匹配避免黑盒 LLM 空间推理）成立。但当前形式在**提示词条件化评估协议的一致性、3D 场景图文献的遗漏与对比、关键设计选择的论证与消融不足、缺真实机器人实验**上有显著问题。判定：**大修（major revision）**。

## ① 3D 场景图文献缺失（[1]3DGraphLLM [2]KeySG [3]DAAAM [4]ConceptGraphs）

> 论文构建了几何导出的 3D 场景图并做语言定位，却未讨论/对比近期大量的 3D 场景图构建与场景图上语言定位文献。Related Work 需要专门讨论这条线，并明确 NS-Grounding 相对它们**做了什么不同/更好**。

**回应（写作）**：Related Work 新增"3D 场景图构建与场景图定位"小节，逐点对比：
- 与 ConceptGraphs（开放词汇场景图 + LLM 推理）的区别：我们显式几何 scorer 打分单射映射（可检查、不依赖黑盒 LLM 推理），且面向**流式部分重建**
- 与 3DGraphLLM/KeySG/DAAAM 的定位
- 措辞模板：*NS-Grounding differs in (i) grounding on streaming/partial reconstructions, (ii) explicit injective graph matching with a hand-crafted geometric scorer library rather than free-form LLM reasoning over scene graphs.*
- 实证锚点见 ⑤（E2 CSVG 对比）

**Response (EN):** We will add a dedicated Related Work subsection on "3D scene-graph construction and language grounding on scene graphs" and contrast NS-Grounding point-by-point. Against ConceptGraphs (open-vocabulary scene graph + LLM reasoning over it), 3DGraphLLM, KeySG, and DAAAM, NS-Grounding differs in three concrete respects: (i) grounding on streaming/partial reconstructions rather than on complete pre-built scans — the scene graph is incrementally grown from the streaming reconstruction, and NS-Grounding must tolerate missing and over-segmented nodes, which none of these methods address; (ii) explicit injective graph matching with a hand-crafted geometric scorer library rather than free-form LLM reasoning over a scene graph, which makes the selection checkable and reproducible instead of relying on a black-box LLM to do spatial reasoning; and (iii) a CLIP visual fallback that binds the symbolic node to the reconstructed instance when the language parse is imperfect. Empirical anchor: see ⑤ (E2 CSVG comparison), where a constraint-based grounding baseline is re-run under our protocol.

## ② label binding / 语义建图对比（[4]ConceptGraphs [5]HOV-SG [6]FunGraph；OVI-MAP [7]）

> 把语义类别集成进 3D 实例建图/融合已是常见做法。请具体解释 "label-mediated feature binding" 与这些方法的区别（不能只是帧率更高），最好有对比；语义 TSDF 建图还应与**OVI-MAP [7]** 对比（问题密切相关）。

**回应**：
- 写作：解释 label binding 机制差异——用类别文本嵌入作为**融合相似度特征**（影响实例合并决策），而非仅贴语义标签；与 ConceptGraphs/HOV-SG/FunGraph 的语义建图（通常事后关联 2D 开放词汇特征）对比
- **E7（OVI-MAP 对比）**：用受控语义绑定消融 + 机制对照，替代直接的 mIoU 并排（口径不同，见下）

**语义绑定消融（同 SAM3 实例 mask、同 CLIP backbone、同 OVI-MAP 协议 eval_per_class_IoU，ScanNet200）**

| 语义绑定方式 | mIoU | mAcc | VLM 查询/实例 |
|---|---|---|---|
| label-mediated（ours） | 0.27* | 0.32* | 0 |
| post-hoc VLM assignment（OVI-MAP-style） | 0.12* | 0.17* | 53* |

- *单场景 scene0025_00 冒烟值；最终表填 30 场景全量均值（30 场景 label-bound arm A 已跑：mIoU 0.3000 / mAcc 0.3915）。
- **机制对照**：OVI-MAP 为 class-agnostic（CropFormer 实体）+ 重建完成后事后挑视角向 VLM 查询投票聚合（其 AQ 列 ≈8.6 次查询/实例）；我们是 SAM3 文本条件掩码 + 2D 帧上标签与 mask 共同产生、随 TSDF 融合累积，每实例 VLM 查询 = 0。
- 因 OVI-MAP 的开放词表 class-agnostic 设定与我们的 prompt-conditioned 设定口径不同，**不并排 mIoU 绝对值**；以本受控消融（同实例/同 CLIP/同协议，只换绑定方式）作为定量证据。

**Response (EN):** We will explain the mechanism precisely: label-mediated binding uses the category text embedding as a **fusion-similarity feature** that participates in instance-merge decisions, not merely as a semantic tag attached after the fact. In our pipeline the label and the mask are produced together in 2D by the same text-conditioned segmenter, and the label's text embedding is bound to the mask *before* TSDF fusion; as a result, semantic evidence accumulates across views during fusion and each instance ends up with a single, consistent, multi-view-averaged semantic feature that also steers whether two fragments merge. This contrasts with ConceptGraphs/HOV-SG/FunGraph, which typically associate open-vocabulary 2D features with the instance *after* reconstruction. For the OVI-MAP comparison we add a controlled semantic-binding ablation (identical SAM3 masks, same CLIP backbone, same OVI-MAP eval_per_class_IoU protocol): label-mediated binding attains mIoU 0.27 / mAcc 0.32 with 0 post-hoc VLM queries per instance, versus 0.12 / 0.17 with ~53 queries for post-hoc VLM assignment (single-scene smoke values; the 30-scene label-bound arm A reaches mIoU 0.3000 / mAcc 0.3915). *Why:* post-hoc assignment must re-project selected views, query a VLM per view, and aggregate by voting — the view selection is noisy and the per-view answers disagree, so it is both more expensive (~53 queries/instance) and less accurate. Because OVI-MAP's open-vocabulary class-agnostic setting differs from our prompt-conditioned setting, we do not place the mIoU absolute values side by side and instead report this controlled ablation as the quantitative evidence, plus the mechanism comparison table as the qualitative evidence.

## ③ ScanNet AP 口径不清

> ScanNet 结果散见论文多处（含摘要），但不清楚 AP 在测什么：3D 实例分割？类无关与否？

**回应（写作）**：全篇统一定义——*class-agnostic 3D instance segmentation AP，IoU@0.5:0.95*；摘要/正文/表注一致；补表注说明评估协议与 GT（ScanNet200 val 官方）。

**Response (EN):** We will standardize the metric definition throughout the paper as *class-agnostic 3D instance segmentation AP at IoU@0.5:0.95*, keep the abstract, main text and table captions consistent, and add a caption stating the evaluation protocol and GT (official ScanNet200 val). This is a class-agnostic instance metric — the segmentation quality of the reconstructed instances independent of their semantic label — which is why it is reported in the reconstruction section, separate from the grounding metrics. We will state this once in the setup and reference it wherever an AP number appears so the reader never has to infer the protocol.

## ④ 每帧全词表 + 运行时随词表规模（[8] OpenGraph/RAM 替代）

> 管线每帧把全词表传给 SAM3（Eq.2）。prior work [8] 更高效：先用开放集识别（如 RAM）提出图像中出现的类别，再只拿缩减集 prompt SAM。请讨论该替代并**报告每帧运行时如何随词表规模扩展**——10 FPS 声明应不独立于 prompt 数。

### E3 — 词表规模 vs 分割帧运行时（实测）

| 词表 V | 每帧延迟 ms | FPS_eff(分割帧) | 峰值显存 MB | segments | oom |
|---|---|---|---|---|---|
| 10 | 311.0 | 3.22 | 16105 | 8.3 | 0 |
| 30 | 508.9 | 1.97 | 16098 | 11.3 | 0 |
| 50 | 761.8 | 1.31 | 16095 | 12.1 | 0 |
| 100 | 1626.6 | 0.61 | 16084 | 16.8 | 0 |
| 200 | | | | | ❌ 待补 |

**回应**：实测显示每分割帧延迟随 V 近线性增长（311→1627ms）——**承认词表成本**；写作讨论 OpenGraph/RAM 两级方案作为未来工作/消融，并说明端到端 FPS 受"每 10 帧一分割"稀释（E5 联动）；⚠️ 待补：显存口径统一 + V=200 + 语义 AP 轴。

**Response (EN):** We measured per-segmentation-frame latency as a function of vocabulary size (E3): 311 ms at V=10, 509 ms at 30, 762 ms at 50, and 1627 ms at 100 — growth that is nearly linear in V, so we acknowledge the vocabulary cost explicitly rather than hiding it behind a single 10 FPS number. *Why it is linear:* SAM3's text-prompted segmentation encodes every prompt in the vocabulary and performs cross-attention between the image and each text prompt, so each additional prompt adds a roughly constant amount of forward computation and memory; the peak-memory column stays essentially flat (16105→16084 MB) while latency grows, confirming the cost is compute-bound per prompt, not memory-bound. We will discuss the two-stage OpenGraph/RAM alternative (open-set detection to preselect the classes that actually appear, then prompt SAM with only the reduced set) as an ablation/future work, and clarify that end-to-end FPS is diluted by the every-10-frames segmentation schedule (see E5), so the 10 FPS claim is stated as a function of prompt count. Remaining: unify the memory-measurement protocol, add V=200, and report a semantic-AP axis (vocabulary size → class coverage → semantic acc).

## ⑤ Sec. III-B.4 七个 scorer 无充分性论证、无消融

### E4 — 7 scorer 逐个消融（99 条均衡子集，无 rerank）

| 配置 | Acc@0.5m@pred | Δ vs full | MeanDist | no_pred |
|---|---|---|---|---|
| **full** | **0.6224** | — | 0.914 | 1 |
| −contact | 0.5612 | −6.1 | 1.050 | 1 |
| −dir | 0.5714 | −5.1 | 1.024 | 1 |
| −vert | 0.5714 | −5.1 | 0.997 | 1 |
| −between | 0.5816 | −4.1 | 1.034 | 1 |
| −rank | 0.5918 | −3.1 | 0.968 | 1 |
| −side | 0.6020 | −2.0 | 0.922 | 1 |
| −near | 0.6735 | +5.1 | 0.899 | 1 |
| −all 7 | 0.3980 | −22.4 | 1.472 | 1 |

关系覆盖：dir 21 / near 16 / rank 18 / vert 15 / contact 16 / between 15 / side 11；parse 失败 0。
**回应**：每个 scorer 边际贡献（除 near 为弱约束伪影，需归因段）+ 全禁 −22.4pt → 必要性成立；关系频率表 → 充分性论证。

**Response (EN):** We add a leave-one-out ablation of the seven scorers (E4, 99-query balanced subset, no rerank). Every scorer contributes a positive margin when removed: −contact −6.1, −dir −5.1, −vert −5.1, −between −4.1, −rank −3.1, −side −2.0 points of Acc@0.5m@pred, and disabling all seven together drops −22.4 points. *Why the pattern:* contact/dir/vert are the largest margins because those relations are both frequent in the query distribution and highly discriminative for disambiguating same-category objects, whereas side is the smallest because it is rarer and often subsumed by dir. The one counter-intuitive result, −near (+5.1), is a weak-constraint artifact: "near" is a soft, low-information relation whose score is often noisy, so removing it removes a misleading signal and occasionally improves top-1 selection; we will attribute this explicitly. The relation-frequency table (dir 21 / near 16 / rank 18 / vert 15 / contact 16 / between 15 / side 11; 0 parse failures) shows every scorer is actually exercised by the query set, which together with the ablation establishes both necessity (each contributes, all-seven removal collapses) and sufficiency (the library covers the relations that occur).

## ⑥ 无真机、无噪声/模糊、无闭环

> 全部实验离线跑 ScanNet200/SceneNN/NR3D/SR3D。无实体机器人演示、无真实传感器噪声/运动模糊评估、无使用定位输出的闭环任务。

### E9 — 合成退化（运动模糊 / 深度噪声 / 丢帧）

| 条件 | AP | AP50 | AP25 |
|---|---|---|---|
| none（复用 e5 基线） | 0.2143 | 0.3947 | 0.6170 |
| blur_w（弱模糊） | 0.2120 | 0.3854 | 0.6087 |
| blur_s（强模糊） | 0.2257 | 0.4202 | 0.6315 |
| blur_xs（极强模糊） | 0.2168 | 0.4069 | 0.5972 |
| noise_w（弱噪声） | 0.2190 | 0.4026 | 0.6324 |
| noise_s（强噪声） | 0.2203 | 0.3987 | 0.6343 |
| noise_xs（极强噪声） | 0.2148 | 0.3861 | 0.6171 |
| drop133（133/200 帧） | 0.1928 | 0.3484 | 0.5443 |
| drop100（100/200 帧） | 0.1696 | 0.3263 | 0.4996 |
| drop67（67/200 帧） | 0.1323 | 0.2760 | 0.4290 |

- 协议：10 场景（scene0025_00/0050_00/0063_00/0064_00/0164_00/0169_00/0193_00/0196_00/0207_00/0249_00），200 帧逐帧分割（--dense-seg），同 e5 基线；合成退化注入 data_eval/scannet 帧序列后重测重建 AP。
- **丢帧单调下降（真实信号）**：drop133→100→67 使 AP 0.1928→0.1696→0.1323（AP25 0.5443→0.4996→0.4290），说明关键帧稀疏化确实损害重建召回。
- **模糊/噪声 ≈ 噪声地板（鲁棒性证据）**：blur/noise 各档 AP 在 0.2120–0.2257（±~0.01，none=0.2143）内非单调抖动，落入 10 场景噪声地板——多视角融合 + FCGF 几何验证对单帧退化不敏感，构成对"无噪声/模糊评估"的正面回应。
- **待补**：对 10 场景做 paired t-test 量化各条件显著性；闭环代理（定位→生成抓取候选/最近邻操作）作为真机替代、并声明局限——真机实验仍列为 future work。

**Response (EN):** We add a synthetic-degradation experiment (E9) injecting motion blur, depth noise, and frame dropping into the data_eval/scannet sequences and re-measuring reconstruction AP (10 scenes, 200-frame per-frame segmentation, same e5 baseline). Three conditions give three distinct behaviors. (1) Frame dropping degrades *monotonically*: drop133→100→67 lowers AP 0.1928→0.1696→0.1323 (AP25 0.5443→0.4996→0.4290). *Why:* fewer keyframes means fewer observations per surface, so the TSDF is sparser and more instances fall below the IoU threshold — this is a genuine, expected failure mode that we now quantify. (2) Depth noise is essentially flat: AP 0.2190/0.2203/0.2148 across weak/strong/extreme noise vs 0.2143 baseline (within ~±0.01). *Why:* TSDF fusion averages the signed-distance observations across ~200 frames, and FCGF geometric verification rejects spurious correspondences, so per-frame depth noise is diluted before it can corrupt instance geometry. (3) Motion blur is likewise within ~±0.01 (0.2120–0.2257), non-monotone, i.e., a noise floor: blur distorts a single view but not the accumulated geometry. Together, (2)+(3) are the *positive* robustness evidence that answers "no noise/blur evaluation". We will add a paired t-test over the 10 scenes to state significance, and a closed-loop proxy (grounding → grasp-candidate / nearest-neighbor manipulation) as a real-robot substitute with stated limitations; physical-robot deployment remains future work.

## ⑦ 符号未定义（Eq.3 的 ID(l_k)）

**回应（写作）**：补 ID(l_k)（l_k 对应实例的标识映射）等全部符号定义。

**Response (EN):** Writing fix — we will define ID(l_k) (the identity mapping that returns the reconstructed instance corresponding to label l_k) and every other symbol in Eq.(3), and cross-check the notation against the implementation so the equation is self-contained and matches the code.

## Minor（写作）

- 错别字/标点：摘要 "RGB-D streams,and"（缺空格）、Sec.I "inference phrase"→"inference phase"、Sec.III-B.4a "Directional Relations.:"（多余句点，多处复发）、Sec.IV-C.a "recall.Additionally"（缺空格）等——全部修正（可用工具自检）。

**Response (EN, Minor):** We will fix all typos/punctuation: abstract "RGB-D streams,and" (missing space), Sec.I "inference phrase"→"inference phase", Sec.III-B.4a "Directional Relations.:" (stray period, recurring), Sec.IV-C.a "recall.Additionally" (missing space), and run an automated spell/grammar check over the full manuscript before resubmission.

---

## References（审稿人提供）

[1] 3DGraphLLM (arXiv:2412.18450) · [2] KeySG (arXiv:2510.01049) · [3] DAAAM (arXiv:2512.00565) · [4] ConceptGraphs (arXiv:2309.16650) · [5] HOV-SG (arXiv:2403.17846) · [6] FunGraph (arXiv:2503.07909) · [7] OVI-MAP (arXiv:2603.26541) · [8] OpenGraph (arXiv:2403.09412)
