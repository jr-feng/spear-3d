# E8 — SceneNN 证据（机制 a：under-segmentation / 词表-标注失配）

> 对应 R1 minor d: "SceneNN 归因无证据"。
> 论文原文: "The performance drop on SceneNN is primarily attributed to the missing
> annotations of small or ambiguous objects ... objects omitted from the task-relevant
> prompt vocabulary are skipped during the streaming stage" → R1: 若类别不在词表内,
> vocabulary=inventory → 损失是设计后果; 需定量支持或列为 limitation。

## 结论（rebuttal 立场）

SceneNN 的 NYU40 标注中有 **38.7% 的实例属于词表外类**
(otherprops 123 inst / 33.8%, otherfurniture 11, television 5, otherstructure 2)，
这些类别在 ScanNet200 词表中**没有任何概念**。由于我们的方法在 streaming 阶段
用词表条件化 SAM3（vocabulary=inventory by design），这些物体**不可能被分割成
标注语义可匹配的实例**——在类感知指标下该类覆盖恒为 0（构造性），在类无关指标下
召回也显著偏低（37.8% vs 62.1% @IoU0.5，词表内物体类；全类 56.7%；仅统计有重建
支撑的 GT 实例）。相比之下，词表内类的实例级匹配召回率健康（62.1% @IoU0.5 /
86.2% @IoU0.25，物体类）。即：SceneNN 差距集中在词表无法表达的 catch-all 类 →
词汇/标注失配（设计后果），方法本身在词表覆盖类上没有失效。**故按 R1 建议列为
limitation，并给出定量证据。**

## 证据 1：词表覆盖率（12 scene GT，E8 step 1）

- 实例级 in-vocab：**61.3%**（out-of-vocab 38.7%）；顶点级 in-vocab：**86.8%**
- 词表外类：otherprops(40) 123 inst = 33.8%、otherfurniture(39) 11、television(25) 5、otherstructure(38) 2
- 工具：scripts/scenenn_vocab_coverage.py → ground_ral/scenenn_vocab_coverage.json
- 说明：NYU40 catch-all 类 + television 在 ScanNet200 概念集无对应词（sofa→couch 等别名已处理）

## 证据 2：类无关实例级匹配（11 scene 重建 vs GT；只统计重建支撑内的 GT 实例）

- vocab-in 类（194 inst）：R@0.25 = **85.1%**，R@0.5 = **56.7%**，R@0.75 = 30.4%
- vocab-in 类（剔除 wall/floor/ceiling 大背景，145 inst）：R@0.25 = **86.2%**，R@0.5 = **62.1%**
- vocab-OUT 类（111 inst，television/otherstructure/otherfurniture/otherprops）：
  R@0.25 = 61.3%，R@0.5 = **37.8%**，R@0.75 = 13.5%
- catch-all only（otherprops 95 + otherfurniture 9 + otherstructure 2 = 106 inst）：
  R@0.25 = 59.4%，R@0.5 = **36.8%**，R@0.75 = 12.3%
- 家具类个例（词表内）：bed 75%、chair 76.9%、sofa 100%、door 83.3%、window 100%、
  pillow 88.9%、clothes 87.5%（@IoU0.5）
- otherprops 类：n=95，R@0.25 = 57.9%，R@0.5 = 34.7%，mean best IoU = 0.359
- 被 support 过滤排除的 GT 实例：vocab-in 2 个，vocab-out 12 个（多为小物体无重建）
- 工具：scripts/scenenn_inst_match.py → ground_ral/scenenn_inst_match.json

关键点：类无关"任意实例覆盖顶点"指标无区分度（98% 全类近满，因实例掩码铺满表面）；
**实例级 IoU 匹配**（是否有"专属掩码"对应 GT 物体）才反映 under-segmentation。
（scripts/scenenn_seg_coverage.py 及 ground_ral/scenenn_seg_coverage.json 为初版"任意掩码
顶点覆盖"指标，全类≈90-98% 无区分度，已被本文件 证据2/3 取代，请勿引用该初版数字。）

## 证据 3：类匹配顶点覆盖（per-NYU40-class class-matched vertex coverage）

GT 类 c 的顶点被"类名映射到 c"的预测实例覆盖的比例（类感知 SceneNN 指标视角）：

- vocab-in 类（全部）：class-matched = **66.9%**（any-mask = 90.1%），gt verts 11.8M
- vocab-in 类（剔除 wall/floor/ceiling，物体类）：class-matched = **61.1%**（any-mask = 96.5%）
- vocab-OUT 类：class-matched = **0.0%**（any-mask = 84.8%）← 构造性：词表无任何词能表达这些 NYU40 类
- 词表内高覆盖个例：door 91.9%、bed 90.3%、window 88.8%、curtain 93.9%、pillow 93.2%、
  night stand 96.0%、box 99.1%、chair 81.3%
- 工具：scripts/scenenn_class_coverage.py → ground_ral/scenenn_class_coverage.json

## 补充证据 4：SceneNN GT 本身的未标注率（"missing annotations" 事实面）

GT 顶点中 class-0（void/未标注）占比按场景平均 **28.2%**
（005: 60.6%、096: 52.6%、263: 50.2%、243: 46.8%、089: 35.0%、080: 24.5%、093: 22.9%、
322: 8.6%、030: 8.9%、011/015: 0%）。"missing annotations of small or ambiguous objects"
在 GT 层面有事实依据；但按 R1 逻辑，词表缺类部分是设计后果 → 列为 limitation。

## Rebuttal 文本（段落草案）

We thank the reviewer for this point. We have quantified the SceneNN deficit and now
state it explicitly as a limitation of our prompt-conditioned design (vocabulary =
inventory). Two numbers summarize the analysis over the 11 SceneNN scenes with
reconstructions (GT statistics over all 12 scenes):
(i) Vocabulary/annotation mismatch is structural: only 61.3% of SceneNN GT instances
belong to NYU40 classes that have any ScanNet200 concept; the remaining 38.7% are
concentrated in NYU40 catch-all classes (otherprops alone 33.8%, plus otherfurniture,
otherstructure, television), which no ScanNet200-derived prompt word can express.
Because our streaming segmenter is conditioned on this vocabulary by design, these GT
objects can never be produced as a semantically matchable instance: their class-matched
vertex coverage is 0.0% by construction, whereas vocabulary-covered object classes reach
61.1% class-matched vertex coverage (66.9% over all covered classes incl. background).
Even under a class-agnostic instance metric (does any dedicated mask match the GT object
at all), out-of-vocabulary instances show markedly lower recall — 37.8% at IoU>=0.5
(61.3% at IoU>=0.25) — versus 56.7%/85.1% for vocabulary-covered classes, i.e. objects
whose names the inventory cannot express are additionally under-segmented during
streaming (no prompt word triggers a mask for them). This is exactly the
vocabulary=inventory design consequence the reviewer describes.
(ii) On the classes the vocabulary CAN express, the method itself is not the bottleneck:
vocabulary-covered object classes reach 62.1% instance-matching recall at IoU>=0.5
(86.2% at IoU>=0.25) and 61.1% class-matched vertex coverage, comparable to the per-class
behaviour we observe on ScanNet200. The SceneNN gap is therefore confined to classes that
are unnameable under our inference-time vocabulary — an annotation/vocabulary mismatch —
rather than a degradation of the streaming reconstruction itself. We have added this as an
explicit limitation in the paper and reported the vocabulary-coverage statistics above
(in Section ...). A superset vocabulary (ScanNet200 ∪ NYU40) would re-cover a large part
of the gap; we note it as future work because SceneNN's NYU40 catch-all classes (e.g.
"otherprops") are not nameable visual concepts for any open-vocabulary segmenter, i.e.
the mismatch is not fully removable by vocabulary engineering alone.
