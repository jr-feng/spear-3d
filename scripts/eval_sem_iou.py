#!/usr/bin/env python3
"""
Vertex-level semantic mIoU / mAcc (ScanNet200 closed-set) — OVI-MAP Table-3 style.

For each scene:
  - load <result_dir>/<scene>/ckpt_final.npz, normalize pred_classes -> ScanNet200 names
  - load GT label txt (id*1000+instance per GT vertex) and GT mesh
  - align GT mesh to recon (utils_3d.align_gt_to_recon, 0.15 m) and map pred masks
    onto GT vertices (get_instances_in_GT_pc), same as evaluate_seqs_scannet
  - per valid GT class c: IoU_c = |gt_c ∩ pred_c| / |gt_c ∪ pred_c|
      mAcc  = mean over classes of |gt_c ∩ pred_c| / |gt_c|
    (background/void GT vertices are excluded from the class set)

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/eval_sem_iou.py \
      --result_dirs /home/OnlineAnySeg/e5_out_200 output_ral/A \
      --gt_dir /home/OnlineAnySeg/data_eval/scannet \
      --scenes scene0011_00,scene0432_00 \
      --tmp /tmp/clsnorm_iou
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

# reuse evaluate's label tables + read/normalize helpers
_real_argv = list(sys.argv)
sys.argv = ["evaluate_seqs_scannet.py", "--result_dir", ".", "--gt_dir", ".",
            "--class-aware"]
import evaluate_seqs_scannet as ev  # noqa: E402
from eval.utils_3d import align_gt_to_recon, get_instances_in_GT_pc  # noqa: E402
sys.argv = _real_argv


def _norm_label(name: str) -> str:
    s = " ".join(str(name or "").strip().lower().split())
    if not s:
        return ""
    if s in ev.LABEL_TO_ID:
        return s
    if s.endswith("s") and s[:-1] in ev.LABEL_TO_ID:
        return s[:-1]
    ALIAS = {
        "trash can": "trash can", "wastebin": "trash can", "trashcan": "trash can",
        "bin": "trash can", "kitchen cabinet": "cabinet", "bathroom cabinet": "cabinet",
        "tv": "television", "couch": "sofa", "sofa": "sofa",
        "coffee table": "table", "dining table": "table", "table": "table",
        "monitor": "monitor", "computer monitor": "monitor",
    }
    if s in ALIAS and ALIAS[s] in ev.LABEL_TO_ID:
        return ALIAS[s]
    return ""


def scene_semantics(npz_path: str, gt_txt: str, gt_ply: str, recon_ply: str):
    """Return (class_ious, class_accs, class_names_present) over GT-valid classes."""
    z = np.load(npz_path, allow_pickle=True)
    # reuse evaluate's reader: label_id/class_name/mask (mask on recon verts)
    pred_info_raw = ev.read_prediction_npz(npz_path)

    gt_ids = np.loadtxt(gt_txt)  # (V_gt,) label*1000+inst
    gt_class = (gt_ids // 1000).astype(int)
    gt_valid_cls = np.isin(gt_class, ev.VALID_CLASS_IDS)

    gt_pc = o3d.io.read_point_cloud(gt_ply)
    recon_pc = o3d.io.read_point_cloud(recon_ply)

    # same alignment path as evaluate_seqs_scannet
    inst_masks = [v["mask"] for v in pred_info_raw.values()]
    inst_masks = np.stack(inst_masks, axis=0)  # (N, V_recon) bool
    valid_recon_pts = np.any(inst_masks, axis=0)
    corr_pt_in_recon, valid_gt_pt_indices = align_gt_to_recon(
        gt_pc, recon_pc, valid_recon_pts_mask=valid_recon_pts, distance_upper_bound=0.15
    )
    pred_info = get_instances_in_GT_pc(pred_info_raw, corr_pt_in_recon,
                                       valid_gt_pt_indices=valid_gt_pt_indices)

    V_gt = len(gt_ids)
    pred_class = np.full(V_gt, -1, dtype=int)
    for key, inst in pred_info.items():
        cls_name = _norm_label(inst.get("class_name") or inst.get("label_id") or "")
        cid = ev.LABEL_TO_ID.get(cls_name, -1)
        if cid not in ev.VALID_CLASS_IDS:
            continue
        m = np.asarray(inst["mask"], dtype=bool)
        if len(m) != V_gt:
            continue
        pred_class[m] = cid

    ious, accs, present = [], [], []
    for c in ev.VALID_CLASS_IDS:
        gt_c = (gt_class == c) & gt_valid_cls
        if gt_c.sum() < ev.opt.min_region_sizes[0]:
            continue
        pred_c = pred_class == c
        inter = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        ious.append(inter / max(int(union), 1))
        accs.append(inter / max(int(gt_c.sum()), 1))
        present.append(ev.ID_TO_LABEL[c])
    return ious, accs, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dirs", nargs="+", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_seg_dir", default=os.path.join(ROOT, "eval", "scannet200", "validation"))
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(d for d in os.listdir(args.result_dirs[0])
                        if os.path.isdir(os.path.join(args.result_dirs[0], d)))

    for rd in args.result_dirs:
        all_iou, all_acc = [], []
        for seq in scenes:
            npz = os.path.join(rd, seq, "ckpt_final.npz")
            ply = os.path.join(rd, seq, "final.ply")
            gf = os.path.join(args.gt_seg_dir, seq + ".txt")
            gp = os.path.join(args.gt_dir, seq, seq + "_vh_clean_2.ply")
            if not all(os.path.isfile(x) for x in (npz, ply, gf, gp)):
                print(f"[skip] {rd}/{seq}")
                continue
            try:
                ious, accs, present = scene_semantics(npz, gf, gp, ply)
            except Exception as exc:  # noqa: BLE001
                print(f"[Failed] {rd}/{seq}: {exc}")
                continue
            if ious:
                print(f"  {seq}: mIoU {np.mean(ious):.4f}  mAcc {np.mean(accs):.4f}  "
                      f"({len(ious)} classes present)")
                all_iou += ious
                all_acc += accs
        if all_iou:
            print(f"=== {rd}: mIoU {np.mean(all_iou):.4f}  mAcc {np.mean(all_acc):.4f} "
                  f"over {len(all_iou)} class-scene entries ===")


if __name__ == "__main__":
    main()
