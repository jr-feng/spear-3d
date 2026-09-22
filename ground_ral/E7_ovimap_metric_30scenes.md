# E7 — OVI-MAP 口径评估记录（30 场景）

## 协议对齐检查（E7 前置；命令来自 ral_readme E7 段）

`compare_ablation.py --result_dirs e5_out_200_dense output_ral/A`（30 场景，A arm，同 GT）

| 口径 | AP | AP50 | AP25 |
|---|---|---|---|
| `e5_out_200_dense`（200 帧均匀子采样 + 逐帧 dense-seg = OVI-MAP 口径） | 0.2135 | 0.3925 | 0.6097 |
| `output_ral/A`（全序列 + 默认 keyframe 设置 = Table I 口径） | 0.2039 | 0.3782 | 0.5835 |
| Δ(e5 − A) | +0.0095 | +0.0143 | +0.0261 |

配对统计（30 场景，scipy.ttest_rel；由粘贴的逐场景值计算，均值与打印值逐位一致）：

| metric | Δ | SE | t | p | e5 胜/负 |
|---|---|---|---|---|---|
| AP | +0.0095 | 0.0061 | 1.55 | 0.131 | 16/14 |
| AP50 | +0.0143 | 0.0112 | 1.28 | 0.212 | 16/14 |
| AP25 | +0.0261 | 0.0122 | 2.14 | **0.041** | 19/10 |

**结论**：两个口径实质等价（胜负 16:14），200 帧 dense 仅略微占优且只在最宽松阈值 AP25 上显著 →
与 OVI-MAP 同口径（200 帧）比较是公平的，不存在"帧数少导致吃亏"的偏置。

## OVI-MAP 官方指标（我们方法，30 场景）

`scripts/ovimap_metric_from_meshes.py`（复用 `/tmp/ovimap30` 已构建语义 mesh，秒级复算）

| 场景集 | mIoU | mAcc |
|---|---|---|
| 30 场景（e3 集，本次） | **0.3057** | **0.3982** |
| 5 场景（scene0011/0432/0527/0559/0689，历史记录） | 0.4197 | 0.5438 |

⚠️ **场景集敏感性大（相差约 27%）**：论文/回复必须固定同一场景集口径，建议统一用 30 场景集，
否则会被质疑挑场景。

## 尚未闭环的缺口（E7 真正交付物）

1. **OVI-MAP 方法自身的数字**（同 30 场景、同 200 帧协议）——目前表里没有 OVI-MAP 行。
   要么复现其算法（重），要么引用其公开数字并明确说明协议/场景集差异（轻，但可能被追问）。
2. `/tmp/ovimap30` 只构建了 `e5_out_200_dense` 一侧的 mesh，**缺 `output_ral/A`**，
   因此全序列口径的 OVI-MAP 指标暂缺。
3. class-aware 指标：`/tmp/clsnorm30` 已有产物（说明跑过），但结果未记录在案。
   注：粘贴里那行 `=== 有效场景平均结果 === AP: 0.2039...` 来自
   `eval/evaluate_seqs_scannet.py:642`，是 compare_ablation 对**最后一个 result_dir（A）**
   打印的**类无关**平均，不是 class-aware 结果。

## 复算命令

```bash
cd /home/OnlineAnySeg
SCENES=$(cat e3_scenes_30.txt | tr '\n' ',' | sed 's/,$//')

# OVI-MAP 官方指标（复用已构建 mesh，秒级）
/root/anaconda3/envs/OASeg/bin/python scripts/ovimap_metric_from_meshes.py \
    --tmp /tmp/ovimap30 --tags _home_OnlineAnySeg_e5_out_200_dense --scenes "$SCENES"

# 若要 A（全序列）侧同口径指标：先构建 mesh，再复算
/root/anaconda3/envs/OASeg/bin/python scripts/ovimap_aligned_eval.py \
    --result_dirs /home/OnlineAnySeg/output_ral/A \
    --gt_dir /home/OnlineAnySeg/data_eval/scannet --scenes "$SCENES" --tmp /tmp/ovimap30
/root/anaconda3/envs/OASeg/bin/python scripts/ovimap_metric_from_meshes.py \
    --tmp /tmp/ovimap30 --tags _home_OnlineAnySeg_output_ral_A --scenes "$SCENES"

# class-aware（记得 tee 保存）
/root/anaconda3/envs/OASeg/bin/python scripts/eval_class_aware.py \
    --result_dirs /home/OnlineAnySeg/e5_out_200_dense output_ral/A \
    --gt_dir /home/OnlineAnySeg/data_eval/scannet --scenes "$SCENES" --tmp /tmp/clsnorm30 \
    2>&1 | tee /tmp/e7_class_aware.log
```
