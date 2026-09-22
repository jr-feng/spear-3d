#!/usr/bin/env python3
"""Export all experiment data recorded in ral_readme.md into one CSV.

Each experiment becomes a section in the CSV, headed by a marker row
"=== E<id> <name> ===", followed by its own header+data rows, then a blank row.
Values are transcribed verbatim from /home/OnlineAnySeg/ral_readme.md.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ground_ral", "experiments_summary.csv")


def section(w, name):
    w.writerow([f"=== {name} ==="])


rows = []

# ---------------- E1 2x2 ablation ----------------
rows.append(("sec", "E1 2×2 消融（SAM3/CropFormer × label-bound/per-crop，分割 AP 类无关）"))
rows.append(("hdr", ["arm", "AP", "AP50", "AP25", "note"]))
rows.append(("data", ["A", "0.2181", "0.3926", "0.6015", "SAM3 + label-bound text（README 原值 21.81/39.26/60.15 %）"]))
rows.append(("data", ["B", "0.2005", "0.3736", "0.5866", "SAM3 + per-crop CLIP"]))
rows.append(("data", ["C", "0.1860", "0.3610", "0.5350", "CropFormer + per-crop（原 OnlineAnySeg，引用值）"]))
rows.append(("data", ["D", "0.2182", "0.3982", "0.5788", "CropFormer + label-bound text"]))
rows.append(("blank", None))

# ---------------- E2 CSVG baseline ----------------
rows.append(("sec", "E2 外部定位基线 CSVG（NR3D/SR3D，qwen-turbo 生成）"))
rows.append(("hdr", ["dataset", "method", "processed", "skipped", "mean_dist", "acc03m_pred", "acc05m_pred",
                     "acc03m_all", "acc05m_all", "coverage", "selacc_all", "selacc_covered",
                     "not_reconstructed", "no_prediction", "wrong_selection", "hallucinated"]))
rows.append(("data", ["NR3D", "CSVG", "1450", "0", "1.099288", "0.544020", "0.574751", "0.451724",
                      "0.477241", "0.962759", "0.455862", "0.473496", "54", "242", "269", "224"]))
rows.append(("data", ["SR3D", "CSVG", "5247", "0", "1.338104", "0.509593", "0.519713", "0.460644",
                      "0.469792", "0.971603", "0.463694", "0.477246", "149", "487", "1074", "1104"]))
rows.append(("blank", None))

# ---------------- E3 vocab scale vs runtime ----------------
rows.append(("sec", "E3 词表规模 vs 运行时（SAM3 分割帧）"))
rows.append(("hdr", ["vocab_V", "median_latency_ms", "FPS_eff", "peak_mem_MB", "segments", "oom"]))
rows.append(("data", ["10", "311.0", "3.22", "16105.3", "8", "0"]))
rows.append(("data", ["30", "508.9", "1.97", "16098.1", "11", "0"]))
rows.append(("data", ["50", "761.7", "1.31", "16094.7", "12", "0"]))
rows.append(("data", ["100", "1626.6", "0.61", "16083.8", "17", "0"]))
rows.append(("blank", None))

# ---------------- E4 scorer ablation ----------------
rows.append(("sec", "E4 scorer 逐个消融（Acc@0.5m@pred）"))
rows.append(("hdr", ["config", "acc05m_pred", "delta_vs_full", "mean_dist"]))
for r in [["full", "0.6224", "—", "0.914"],
          ["-contact", "0.5612", "-6.1", "1.050"],
          ["-dir", "0.5714", "-5.1", "1.024"],
          ["-vert", "0.5714", "-5.1", "0.997"],
          ["-between", "0.5816", "-4.1", "1.034"],
          ["-rank", "0.5918", "-3.1", "0.968"],
          ["-side", "0.6020", "-2.0", "0.922"],
          ["-near", "0.6735", "+5.1", "0.899"],
          ["-all7", "0.3980", "-22.4", "1.472"]]:
    rows.append(("data", r))
rows.append(("blank", None))

# ---------------- E4 relation coverage ----------------
rows.append(("sec", "E4 关系覆盖分析 rel_type 频率（99 查询，parse 失败 0，无关系 0）"))
rows.append(("hdr", ["rel_type", "count", "pct_of_queries"]))
for r in [["near", "16", "16.2"], ["between", "15", "15.2"], ["closest_to", "13", "13.1"],
          ["above", "12", "12.1"], ["same_side_as", "11", "11.1"], ["on_top_of", "9", "9.1"],
          ["right_of", "8", "8.1"], ["under", "7", "7.1"], ["front_of", "6", "6.1"],
          ["behind", "6", "6.1"], ["farthest_from", "5", "5.1"], ["below", "3", "3.0"],
          ["not_with_object", "1", "1.0"], ["on_wall", "1", "1.0"], ["left_of", "1", "1.0"],
          ["with_object", "1", "1.0"], ["has_on_it", "1", "1.0"]]:
    rows.append(("data", r))
rows.append(("blank", None))

rows.append(("sec", "E4 scorer 覆盖（查询触及 ≥1 条该 scorer 的关系）"))
rows.append(("hdr", ["scorer", "queries", "pct_of_queries"]))
for r in [["dir", "21", "21.2"], ["near", "16", "16.2"], ["rank", "18", "18.2"],
          ["vert", "15", "15.2"], ["contact", "16", "16.2"], ["side", "11", "11.1"],
          ["between", "15", "15.2"]]:
    rows.append(("data", r))
rows.append(("blank", None))

# ---------------- E5 timing ----------------
rows.append(("sec", "E5 端到端 FPS（RTX 3090，同 5 场景）"))
rows.append(("hdr", ["scene", "A_arm_SAM3_FPS", "C_arm_CropFormer_FPS"]))
for r in [["scene0432_00", "11.42", "5.50"], ["scene0527_00", "10.71", "4.82"],
          ["scene0559_00", "10.62", "4.41"], ["scene0494_00", "10.82", "6.60"],
          ["scene0689_00", "11.04", "5.25"], ["mean", "10.92", "5.32"]]:
    rows.append(("data", r))
rows.append(("blank", None))

rows.append(("sec", "E5 per-stage 吞吐（各阶段独立 fps）"))
rows.append(("hdr", ["stage", "A_arm_SAM3", "C_arm_CropFormer"]))
for r in [["分割", "4.16 call/s (avg 241 ms)", "0.89 frame/s (avg 1099 ms)"],
          ["融合", "3.03 frame/s (avg 330 ms)", "2.41 frame/s (avg 453 ms)"],
          ["合并", "2.05 merge/s (avg 487 ms)", "1.50 merge/s (avg 1307 ms)"]]:
    rows.append(("data", r))
rows.append(("blank", None))

# ---------------- E7 OVI-MAP semantics ----------------
rows.append(("sec", "E7 OVI-MAP 语义对比（eval_per_class_IoU，30 场景，label-bound arm A）"))
rows.append(("hdr", ["arm", "scenes", "mIoU", "mAcc"]))
rows.append(("data", ["A label-bound", "30", "0.3000", "0.3915"]))
rows.append(("blank", None))

# ---------------- E8 SceneNN summary ----------------
rows.append(("sec", "E8 SceneNN 证据（词表覆盖/实例匹配召回/类匹配顶点覆盖，11 场景，054 无 ckpt）"))
rows.append(("hdr", ["metric", "value"]))
for r in [["GT 实例词表内覆盖率（12 scene）", "61.3%"],
          ["GT 顶点词表内覆盖率", "86.8%"],
          ["vocab-in 实例匹配召回 @IoU0.5 / @0.25", "56.7% / 85.1%"],
          ["vocab-out 实例匹配召回 @IoU0.5 / @0.25", "37.8% / 61.3%"],
          ["vocab-in 类匹配顶点覆盖（物体类）", "66.9%（61.1%）"],
          ["vocab-out 类匹配顶点覆盖", "0.0%（构造性）"],
          ["GT 未标注（class-0）顶点占比（场景平均）", "~28%"],
          ["词表外实例占比（otherprops 等）", "38.7%（otherprops 33.8%）"]]:
    rows.append(("data", r))
rows.append(("blank", None))

# ---------------- E9 robustness ----------------
rows.append(("sec", "E9 退化鲁棒性（10 场景，200 帧逐帧分割，none 复用 e5 基线）"))
rows.append(("hdr", ["condition", "AP", "AP50", "AP25"]))
for r in [["none", "0.2143", "0.3947", "0.6170"],
          ["blur_w", "0.2120", "0.3854", "0.6087"],
          ["blur_s", "0.2257", "0.4202", "0.6315"],
          ["blur_xs", "0.2168", "0.4069", "0.5972"],
          ["noise_w", "0.2190", "0.4026", "0.6324"],
          ["noise_s", "0.2203", "0.3987", "0.6343"],
          ["noise_xs", "0.2148", "0.3861", "0.6171"],
          ["drop133", "0.1928", "0.3484", "0.5443"],
          ["drop100", "0.1696", "0.3263", "0.4996"],
          ["drop67", "0.1323", "0.2760", "0.4290"]]:
    rows.append(("data", r))
rows.append(("blank", None))

# ---------------- E10 lambda summary ----------------
rows.append(("sec", "E10 λu 重扫（NR3D，qwen3，100 条）"))
rows.append(("hdr", ["lambda_u", "MeanDist_none", "Acc03_none", "Acc05_none",
                     "MeanDist_norm", "Acc03_norm", "Acc05_norm"]))
for r in [["0.00", "1.4601", "0.4500", "0.4700", "1.4601", "0.4500", "0.4700"],
          ["0.25", "1.4251", "0.4600", "0.4800", "1.3090", "0.4800", "0.5000"],
          ["0.50", "1.2048", "0.5200", "0.5400", "1.1976", "0.5100", "0.5300"],
          ["0.75", "1.2157", "0.5000", "0.5200", "1.2406", "0.5300", "0.5400"],
          ["1.00", "1.6963", "0.3600", "0.3700", "1.6963", "0.3600", "0.3700"]]:
    rows.append(("data", r))
rows.append(("blank", None))

rows.append(("sec", "E10 项尺度诊断（median s_node vs s_rel，qwen3 100 条）"))
rows.append(("hdr", ["metric", "value"]))
for r in [["median s_node", "0.951"], ["median s_rel", "0.371"], ["gap", "0.579"],
          ["s_node ≥0.9 占比", "64.6%"], ["s_rel ≤0.5 占比", "62.6%"]]:
    rows.append(("data", r))
rows.append(("blank", None))

rows.append(("sec", "E10 决策变化率（norm vs none，同 λu，top-1）"))
rows.append(("hdr", ["lambda_u", "changed", "total", "pct"]))
for r in [["0.25", "5", "100", "5.0%"], ["0.50", "9", "100", "9.0%"], ["0.75", "10", "100", "10.0%"]]:
    rows.append(("data", r))
rows.append(("blank", None))

rows.append(("sec", "E10 骨干稳健性（λu=0.5，NR3D）"))
rows.append(("hdr", ["backbone", "Acc05_none", "Acc05_norm", "Acc03_none", "Acc03_norm"]))
for r in [["qwen3", "0.5400", "0.5300", "0.5200", "0.5100"],
          ["deepseek", "0.5758", "0.5657", "0.5354", "0.5253"],
          ["glm", "0.5400", "0.5400", "0.5200", "0.5100"]]:
    rows.append(("data", r))


with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    for kind, payload in rows:
        if kind == "sec":
            w.writerow([payload])
        elif kind == "hdr":
            w.writerow(payload)
        elif kind == "data":
            w.writerow(payload)
        elif kind == "blank":
            w.writerow([])

print(f"saved -> {OUT}")
print(f"rows written: {len(rows)}")
