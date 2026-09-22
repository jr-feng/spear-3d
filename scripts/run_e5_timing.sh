#!/usr/bin/env bash
# =============================================================================
# E5 — per-stage timing, both pipelines, same hardware/keyframe settings
#
#   A-arm (SAM3 + label-bound)  : main_eval_scannet.py (built-in E5 timing)
#   C-arm (CropFormer / OAS)    : main.py (patched E5-OAS timing)
#
# Both use keyframe_freq=10, seg_add_interval=10, merge_frame_interval=50
# (check the two configs: config/scannet_test_1024.yaml vs config/scannet_cropformer.yaml)
# on the same scenes, so the per-stage numbers are directly comparable.
#
# NOTE on the "4.6 FPS" claim: the OAS line below reports ONLINE stages only
# (integrate / insert_seg / merge / other). CropFormer mask prediction is a
# separate offline stage (scripts/mask_predict/...), same as the OAS paper's
# "15 FPS" being the merge stage only. To answer R1 "15 vs 4 FPS" we report
# BOTH: (i) online-stage FPS, and (ii) CropFormer segmentation throughput,
# and the sum gives the true end-to-end figure.
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/root/anaconda3/envs/OASeg/bin/python

# ---- scenes (same set as the existing E5 SAM3 table) ----
SCENES="${SCENES:-scene0432_00 scene0527_00 scene0559_00 scene0494_00 scene0689_00}"

# ---- A arm: SAM3 + label-bound (uses SAM3 service via tunnel 18090) ----
echo "##################################################################"
echo " E5 timing — ARM A (SAM3 + label-bound text)"
echo "##################################################################"
$PY "$ROOT/main_eval_scannet.py" --arm A \
    --config "$ROOT/config/scannet_test_1024.yaml" \
    --scans-root "$ROOT/data_eval/scannet" \
    --device cuda:0 --merge-gpu 0 \
    -o "$ROOT/e5_timing_A" \
    --scenes $SCENES \
    2>&1 | tee "$ROOT/e5_timing_A.log"

echo
echo "##################################################################"
echo " E5 timing — ARM C (CropFormer / OAS backend)"
echo "##################################################################"
# C arm runs the OAS backend (main.py) with pre-computed CropFormer masks.
# The masks must already exist under the instance_dir the dataset expects;
# if not, generate them first with scripts/mask_predict/mask_predict_single_seq_w_semantic.py.
# Loop scenes manually because main.py takes one --seq_name at a time.
for SEQ in $SCENES; do
    echo "--- scene $SEQ ---"
    $PY "$ROOT/main.py" \
        -c "$ROOT/config/scannet_cropformer.yaml" \
        -d "$ROOT/data_eval/scannet/$SEQ" \
        -i "$CROPFORMER_INSTANCE_DIR/${SEQ}" \
        --seq_name "$SEQ" \
        --device cuda:0 \
        -o "$ROOT/e5_timing_C" \
        2>&1 | tee -a "$ROOT/e5_timing_C.log" | grep -E "E5-OAS|Input sequence finished"
done

echo
echo "Done. Logs: e5_timing_A.log (arm A)  e5_timing_C.log (arm C)"
echo "Extract the [E5 per-stage] / [E5-OAS] lines into the rebuttal table."
