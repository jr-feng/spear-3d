# archive/ — 旧实验脚本与备份归档

本目录存放**不再用于当前 SPEAR-3D 主流程**的旧实验脚本、早期版本与备份，
统一归档以便根目录保持整洁。**实验数据（`output_test/`、`models/`、数据集目录）未移动。**

## 当前主流程（保留在根目录，勿动）

| 入口 | 用途 |
|---|---|
| `main_eval_scannet.py` | ScanNet200 流式重建（SAM3 掩膜 + OnlineAnySeg 后端） |
| `main_eval_sceneNN.py` | SceneNN 流式重建（同上） |
| `LLM_sam3_query_graph_clip.py` | NS-Grounding：LLM 解析查询图 + CLIP 类匹配 + 图匹配定位（NR3D/SR3D） |

## 归档内容清单

### 早期 LLM 定位脚本（当前版本为 `LLM_sam3_query_graph_clip.py`）
- `LLM_sam3.py` — 最早的 LLM 直出方案
- `LLM_sam3_dist.py` — 距离评分变体
- `LLM_sam3_geo.py` — 几何评分变体
- `LLM_sam3_graph_dist.py` — 场景图 + 距离评分
- `LLM_sam3_query_graph_clip_edge.py` — 查询图 + 边评分变体（含 `EdgeGroundingConfig`/`EdgeQueryGraphGrounder`）

### 早期入口/基线脚本
- `main.py` — OnlineAnySeg 官方入口（git mv 保留历史；README 中的 `python main.py -c ...` 用法对应此文件）
- `main_crop.py` — CropFormer 入口（旧基线）
- `main_eva_2.py` / `main_eval.py` — 早期评估入口
- `main_test.py` / `main_test_2.py` — 早期测试入口（`main_test_2.py` 依赖 `scene_generator.py`、`instance_tracker.py`，已一并归档）

### 工具/备份
- `log.py` / `log_1.py` — 日志工具（早期版本，未被当前入口引用）
- `scene_update_0.py` — `scene_update.py` 的早期备份
- `check_npz.py` / `inspect_ckpt_final.py` — checkpoint 检查工具
- `read_depth.py` / `scene_generator.py` / `server.py` — 零散工具（`server.py` 为本地推理服务脚本，与 `sam3_api.py` 调用的远端 SAM3 服务无关）
- `instance_tracker.py` — 在线实例跟踪模块（未被当前入口引用）
- `deploy_cpp/` — OnlineAnySeg 的 C++ 部署组件（未被 `voxel_hashing.py` 引用）
- `scanrefer.zip` — ScanRefer 数据压缩包备份（数据集本体在 `scanrefer/`）

### 论文文件（移至 `paper/`）
- `main.tex`、`(v1)SPEAR_3D_*.pdf`、`paper_text.txt`（PDF 提取文本）、`generate_query_graph_paper_figures.py`（插图生成，已加 `sys.path` 使可从 `paper/` 目录运行）

## 如何运行归档脚本

归档脚本需要从仓库根目录导入模块（扁平 import 结构）。两种方式：

```bash
# 方式一：从仓库根目录运行，设置 PYTHONPATH
cd /home/OnlineAnySeg && PYTHONPATH=. python archive/<script>.py ...

# 方式二：临时移回根目录运行后移回
```

## 已删除的临时/垃圾文件（2026-08-20 整理时）

`__pycache__/`、`Gemini_Generated_Image_*.png`、`depth_vis_*.png`、`test.py`（317B）、
`llm_outputs.jsonl`（17B）、`coco/`（空目录）。
