#!/usr/bin/env bash
# =============================================================================
# E9 退化鲁棒性实验 — 一键运行（9 退化档 + none 基线）
#
#   阶段:
#     gen    生成 9 个退化帧副本（CPU，分钟级）
#     recon  逐档流式重建（GPU + SAM3 隧道，唯一长任务，断点续跑）
#     eval   class-agnostic / class-aware / OVI-MAP 三口径评估（CPU）
#     report 汇总 markdown 表（none 行自动复用 e5_out_200_dense 基线）
#
#   退化档（轻/强/极强）:
#     blur    运动模糊 核长 7 / 21 / 41 px
#     noise   深度噪声 b=0.005/0.02/0.04 (sigma_mm=0.5+b*d_m^1.5), 洞 3%/10%/20%
#     drop    均匀丢帧 133(6.7Hz) / 100(5Hz) / 67(3.3Hz)
#
# 用法:
#   bash scripts/run_e9.sh                      # 全流程
#   bash scripts/run_e9.sh gen                  # 只生成副本
#   SCENES="scene0025_00,scene0050_00" bash scripts/run_e9.sh recon   # 只重建指定场景
#
# 环境变量可覆盖:
#   SCENES, DEGRADE_ROOT, OUT_ROOT, CONFIG, DEVICE, MERGE_GPU
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/root/anaconda3/envs/OASeg/bin/python

SCENES="${SCENES:-scene0025_00,scene0050_00,scene0063_00,scene0064_00,scene0164_00,scene0169_00,scene0193_00,scene0196_00,scene0207_00,scene0249_00}"
DEGRADE_ROOT="${DEGRADE_ROOT:-$ROOT/data_eval_robust}"
OUT_ROOT="${OUT_ROOT:-$ROOT/out_robust}"
CONFIG="${CONFIG:-$ROOT/config/scannet_test_1024.yaml}"
DEVICE="${DEVICE:-cuda:0}"
MERGE_GPU="${MERGE_GPU:-0}"

# 9 个退化档（gen/recon 只跑这些）；none 在 eval/report 时加进来并自动映射 e5 基线
DEG_CONDS="blur_w,blur_s,blur_xs,noise_w,noise_s,noise_xs,drop133,drop100,drop67"
ALL_CONDS="none,${DEG_CONDS}"

STAGE="${1:-all}"

cd "$ROOT"

run_stage() {
  echo
  echo "=============================================================="
  echo " E9 stage: $1"
  echo "=============================================================="
  shift
  "$PY" scripts/e9_robustness.py "$@"
}

case "$STAGE" in
  gen)
    run_stage gen --stage gen --scenes "$SCENES" --conditions "$DEG_CONDS" --degrade-root "$DEGRADE_ROOT"
    ;;
  recon)
    run_stage recon --stage recon --scenes "$SCENES" --conditions "$DEG_CONDS" \
      --degrade-root "$DEGRADE_ROOT" --out-root "$OUT_ROOT" \
      --config "$CONFIG" --device "$DEVICE" --merge-gpu "$MERGE_GPU" \
      2>&1 | tee /tmp/e9_recon_v2.log
    ;;
  eval)
    run_stage eval --stage eval --scenes "$SCENES" --conditions "$ALL_CONDS" --out-root "$OUT_ROOT"
    ;;
  report)
    run_stage report --stage report --scenes "$SCENES" --conditions "$ALL_CONDS" --out-root "$OUT_ROOT"
    ;;
  all)
    run_stage gen --stage gen --scenes "$SCENES" --conditions "$DEG_CONDS" --degrade-root "$DEGRADE_ROOT"
    run_stage recon --stage recon --scenes "$SCENES" --conditions "$DEG_CONDS" \
      --degrade-root "$DEGRADE_ROOT" --out-root "$OUT_ROOT" \
      --config "$CONFIG" --device "$DEVICE" --merge-gpu "$MERGE_GPU" \
      2>&1 | tee /tmp/e9_recon_v2.log
    run_stage eval --stage eval --scenes "$SCENES" --conditions "$ALL_CONDS" --out-root "$OUT_ROOT"
    run_stage report --stage report --scenes "$SCENES" --conditions "$ALL_CONDS" --out-root "$OUT_ROOT"
    ;;
  *)
    echo "用法: bash scripts/run_e9.sh [gen|recon|eval|report|all]  (默认 all)" >&2
    exit 2
    ;;
esac

echo
echo "E9 完成。产物:"
echo "  退化副本 : $DEGRADE_ROOT/<cond>/"
echo "  重建结果 : $OUT_ROOT/<cond>/"
echo "  汇总表   : $OUT_ROOT/e9_summary.md"
