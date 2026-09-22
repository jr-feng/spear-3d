#!/usr/bin/env bash

set -uo pipefail

OAS_ROOT="${1:-output_test/scannet_sam3}"
GT_ROOT="${2:-data_eval/scans}"
LOG_DIR="${3:-scanrefer_oas_eval/sam3_semantic}"

mkdir -p "${LOG_DIR}"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
run_log="${LOG_DIR}/run_scannet_add_classes_and_mask_iou_${timestamp}.log"
fail_log="${LOG_DIR}/run_scannet_add_classes_and_mask_iou_${timestamp}.failed.log"

if [[ ! -d "${OAS_ROOT}" ]]; then
  echo "[Error] OAS root not found: ${OAS_ROOT}" | tee -a "${run_log}"
  exit 1
fi

if [[ ! -d "${GT_ROOT}" ]]; then
  echo "[Error] GT root not found: ${GT_ROOT}" | tee -a "${run_log}"
  exit 1
fi

shopt -s nullglob
scene_dirs=("${OAS_ROOT}"/*)

total=0
success=0
failed=0

echo "[Info] OAS root: ${OAS_ROOT}" | tee -a "${run_log}"
echo "[Info] GT root: ${GT_ROOT}" | tee -a "${run_log}"
echo "[Info] Log file: ${run_log}" | tee -a "${run_log}"
echo "[Info] Fail log: ${fail_log}" | tee -a "${run_log}"

for scene_dir in "${scene_dirs[@]}"; do
  [[ -d "${scene_dir}" ]] || continue

  scene="$(basename "${scene_dir}")"
  total=$((total + 1))

  echo "" | tee -a "${run_log}"
  echo "[Scene ${total}] ${scene}" | tee -a "${run_log}"

  if ! python scanrefer_oas_eval/add_classes_to_pred_info.py \
    --scene-id "${scene}" \
    --oas-root "${OAS_ROOT}" \
    --gt-root "${GT_ROOT}" >> "${run_log}" 2>&1; then
    failed=$((failed + 1))
    echo "${scene},add_classes_to_pred_info" | tee -a "${fail_log}" "${run_log}"
    continue
  fi

  if ! python scanrefer_oas_eval/map_pred_to_gt_mask_iou.py \
    --scene-id "${scene}" \
    --oas-root "${OAS_ROOT}" \
    --gt-root "${GT_ROOT}" >> "${run_log}" 2>&1; then
    failed=$((failed + 1))
    echo "${scene},map_pred_to_gt_mask_iou" | tee -a "${fail_log}" "${run_log}"
    continue
  fi

  success=$((success + 1))
  echo "[OK] ${scene}" | tee -a "${run_log}"
done

echo "" | tee -a "${run_log}"
echo "[Summary] total=${total} success=${success} failed=${failed}" | tee -a "${run_log}"

if [[ "${failed}" -gt 0 ]]; then
  echo "[Summary] failed scenes recorded in ${fail_log}" | tee -a "${run_log}"
fi
