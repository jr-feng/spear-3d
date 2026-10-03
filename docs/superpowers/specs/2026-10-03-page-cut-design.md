# 正文压到 8 页

日期：2026-10-03

范围：Overleaf 上的 SPEAR-3D 正文，以及补充材料 `paper/supplementary.tex` 的本地副本和 Overleaf 副本。目标是整篇正文 PDF（含参考文献）不超过 8 页。四张正文表全部留下。式 (9) 的数学不改。图 1 不改。不重跑实验，不改 `GRAPH_BALANCE`，不填写视频空白。

本规格取代 `2026-10-03-manuscript-correspondence-design.md` 第 4 节里“不移动图 4、图 6、图 8”和第 7 节里“先缩短相关工作、再移走 CSVG”的规定。表体、公式含义、指标定义和视频空白仍以那份规格为准。

## 1. 图

当前正文的图按出现顺序是：图 1 结构框图，图 2 `struct.pdf`，图 3 `stream_update.pdf`，图 4 `pred_xy.pdf`（`fig:final_xy`），图 5 between 的 TikZ（`fig:query_graph`），图 6 `scene_graph.pdf`（`fig:scene_graph`），图 7 `rank.pdf`（`fig:rerank_score`），图 8 `pred_3d.pdf`（`fig:final_3d`）。

图 4、图 6、图 8 这三张里，正文只留图 4。图 4 是俯视图，说明 between 约束怎样排除干扰物，定性分析引用它。图 6 是场景图拓扑，图 8 是同一例子的三维框。这两张移到补充材料。图 1、图 2、图 3、图 5、图 7 仍留在正文。

`pred_xy.pdf` 的活动图环境留在现在的位置，也就是重建后端、体素投影公式之后。不要把它挪到定性分析。它现在排在图 5 之前；挪走之后，回信里的“图 5 / Fig.~5”会不再指向 between 图。

正文删除 `scene_graph.pdf` 和 `pred_3d.pdf` 两个图环境。更新段落里被注释掉的第二份 `pred_xy` 图也删掉，使 `fig:final_xy` 只定义一次。

补充材料在“Robot Demonstration Video”之前新增一节 Additional Figures。该节之前设置 `\renewcommand{\thefigure}{S\arabic{figure}}`。表号计数器不动，表仍是 S1–S10。两张图按这个顺序放入，题注原文保留：

1. `scene_graph.pdf`，题注 “3D scene graph topology. The graph encodes explicit spatial relations as the substrate for structural graph matching.”
2. `pred_3d.pdf`，题注 “Final 3D spatial localization. The red bounding box indicates the predicted target extent in the reconstructed scene.”

节首加一句：These two figures use the same chair example as Figure 4 of the manuscript. 补充材料与正文不是同一次编译，正文不使用 `\ref` 指向这两张图。

去掉图 6 和图 8 之后，between 图仍是图 5。`rank.pdf` 会变成图 6，正文对它的引用继续用 `\ref{fig:rerank_score}`。

## 2. 方法章

公式全部留下，包括房间侧向、between、观察方向上的最大化，以及式 (9)。每个关系段已经是一句定义加公式，这些段落不删。

打分器库开头那段符号定义留下。其中有场景自适应尺度、\(\gamma_{xy}\)、\(\rho\)、\(r^{\star}\)、\(a_{xy}\)、\(g_z\)、\(\mathrm{ID}(l_k)\)、\(\mathcal{U}\)、\(\Pi\)、\(\mathcal{R}\)、\(\sigma\) 和 \(\phi\)。这些句子是前面公式的定义，不当作重复叙述删掉。

只删这两句：

- 融合分数公式之后的 “The resulting voxelized observations and matching scores are then passed to our category-guided streaming instance update module.”
- 实例更新公式之后的 “The streaming reconstruction stage runs at approximately 10 FPS on an RTX 3090.”

式 (9) 后面的权重定义、束搜索宽度 160、target limit 24、reference limit 10、class top-k 3 都留下。不改式 (9) 的源码，包括其中的 `\resizebox`。

## 3. 实验章

四张表不删、不改数字、不改表题。被注释掉的旧重建表留着，它不进 PDF。

### 3.1 实验设置

第二段改成：

All experiments use an RTX 3090. NS-Grounding is called on demand and takes about 2 seconds per query with the Qwen3-plus API. Reconstruction throughput and query latency are reported separately.

第一段不改。

### 3.2 超参数

整段换成：

The grounding score is the weighted geometric mean in \eqref{eq:graph_score}, with \(\lambda_u=0.50\). Node weights are multiplied by \(\lambda_u\) and relation weights by \(1-\lambda_u\).

### 3.3 重建性能

整段换成：

On an RTX 3090 with keyframe interval 10, the proposed arm runs at 10.92 FPS end to end (Table~\ref{tab:seg_results}). The 4 FPS entry for OnlineAnySeg in that table is transcribed from OnlineAnySeg, whose published figure of about 15 FPS was measured on an RTX 4090. On ScanNet200 the primary metric is class-agnostic AP, which is 22.6; AP\(_{25}\) is 60.4. On SceneNN the method is below OnlineAnySeg on AP, AP\(_{50}\), and AP\(_{25}\), and below MaskClustering on AP and AP\(_{50}\), because classes absent from the prompt list receive no class-matched credit. EmbodiedSAM is higher on ScanNet200 because it is trained on that dataset; it is closed-set.

### 3.4 定位性能

删掉段首两句（从 “In the supplementary material” 到 “improve target localization.”）。从 “Acc@0.3m and Acc@0.5m are centroid distances” 起整段留下，包括 58.8、67.0、65.9 不是 SR3D 最高值，以及 deepseek-v3.2 的 SR3D T-Acc 64.0 / 57.0 高于 GLM-5 的 63.2 / 53.8。

### 3.5 消融

整段换成：

The supplementary material compares each LLM with and without NS-Grounding. On every backbone, NS-Grounding raises Acc@0.5m and both T-Acc columns on NR3D and on SR3D.

### 3.6 \(\lambda_u\)

整段换成：

The supplementary material reports \(\lambda_u \in \{0.25, 0.50, 0.75\}\) on NR3D with Qwen3-plus. Acc@0.5m is highest at \(\lambda_u=0.50\).

### 3.7 定性分析

三个步骤换成这一段，`rank.pdf` 的图环境留在原处：

For the instruction ``choose the chair that is in the center of the plant and the radiator,'' Fig.~\ref{fig:query_graph} shows one ternary between factor on the chair and the two anchors. Fig.~\ref{fig:rerank_score} shows that several chairs match the category, while chair \#5 receives the highest geometric score; the black-box LLM baseline selects distractor chair \#3. Fig.~\ref{fig:final_xy} shows the selected chair in the XY plane, aligned with the ground-truth target. The scene-graph topology and the 3D localization of the same example are in the supplementary material.

## 4. 回信

不改两封回信。图 5 的编号不变，回信里的 “Fig.~5” 和 “图 5” 仍然指向 between 图。实施时若发现回信用数字引用了图 6、图 7 或图 8，再把那一处改成补充材料中的对应图；当前两封回信没有这种引用。

## 5. 若仍超过 8 页

先按第 1–3 节改完并重新编译。若 PDF 仍是 9 页，只再缩短打分器库开头那段：删掉 “In the following, \(s_{\mathrm{dir}}\), ... denote the scores of various correspondence relationships.” 这一句。尺度和符号定义仍留下。

这一次之后若仍超过 8 页，停止。不删表，不删图 1、图 4、图 5、图 7，不改公式，不缩小到 `\footnotesize` 以下，不拆数字。

## 6. 完成标准

- Overleaf 上 SPEAR 正文仍是主文件。补充材料不是主文件。
- 重新编译正文：错误数为 0，PDF 页数小于或等于 8。
- 正文仍有 `fig:final_xy`、`fig:query_graph`、`fig:rerank_score`，以及四张表的标签。正文不再出现 `scene_graph.pdf` 或 `pred_3d.pdf`。
- 本地 `paper/supplementary.tex` 与 Overleaf 上的 `supplementary.tex` 都含这两张图，图号为 S1 和 S2。视频处的方框仍是 “[Video file to be inserted here]”。
- 不编辑 `paper/main.tex`。不提交除本规格以外的仓库改动，除非另有要求。
