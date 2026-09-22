#!/usr/bin/env python3
"""
OVI-MAP-style class-aware instance evaluation (AP25/AP50/AP75).
Predicted instances must match the GT class (ScanNet200) AND the IoU threshold
to count as correct; per-scene results averaged over the requested scenes.

Pipeline per scene:
  1. load <result_dir>/<scene>/ckpt_final.npz
  2. normalize pred_classes text (SAM3 outputs e.g. "doors") -> canonical
     ScanNet200 class name (lower/strip/singular + alias map)
  3. write a temporary normalized npz, feed it to evaluate_seqs_scannet with
     --class-aware (the no_class remap is disabled, so real labels are matched)

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/eval_class_aware.py \
      --result_dirs output_ral/A output_ral/B \
      --gt_dir /home/OnlineAnySeg/data_eval/scannet \
      --scenes scene0011_00,scene0015_00 \
      --tmp /tmp/clsnorm
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'eval'))

# ---- import evaluate (class-aware flag is read at import time) ----
_real_argv = list(sys.argv)
sys.argv = ["evaluate_seqs_scannet.py", "--result_dir", ".", "--gt_dir", ".",
            "--class-aware"]
import evaluate_seqs_scannet as ev  # noqa: E402
sys.argv = _real_argv  # restore for this script's own argparse


def _norm_label(name: str) -> str:
    """Normalize a raw class text to a canonical ScanNet200 label name."""
    s = " ".join(str(name or "").strip().lower().split())
    if not s:
        return ""
    if s in ev.LABEL_TO_ID:
        return s
    # strip trailing 's' (doors -> door) when the singular exists
    if s.endswith("s") and s[:-1] in ev.LABEL_TO_ID:
        return s[:-1]
    # alias map for common mismatches
    ALIAS = {
        "trash can": "trash can", "wastebin": "trash can", "trashcan": "trash can",
        "bin": "trash can", "kitchen cabinet": "cabinet", "bathroom cabinet": "cabinet",
        "tv": "television", "couch": "sofa", "sofa": "sofa",
        "coffee table": "table", "dining table": "table", "table": "table",
        "monitor": "monitor", "computer monitor": "monitor",
    }
    if s in ALIAS and ALIAS[s] in ev.LABEL_TO_ID:
        return ALIAS[s]
    return ""  # unmatched -> instance dropped (reported below)


def normalize_scene_npz(npz_path: str, out_path: str) -> int:
    z = np.load(npz_path, allow_pickle=True)
    content = {k: z[k] for k in z.files}
    raw = list(content["pred_classes"])
    mapped = []
    unmatched = 0
    for c in raw:
        n = _norm_label(c)
        if not n:
            n = str(c).strip().lower()
            unmatched += 1
        mapped.append(n)
    content["pred_classes"] = np.array(mapped)  # fixed-width <U array (read_prediction_npz uses np.load w/o allow_pickle)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **content)
    return unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dirs", nargs="+", required=True)
    ap.add_argument("--gt_dir", required=True, help="dir with <scene>/<scene>_vh_clean_2.ply")
    ap.add_argument("--gt_seg_dir", default=os.path.join(ROOT, "eval", "scannet200", "validation"))
    ap.add_argument("--scenes", default=None, help="comma-separated; default = first result_dir scenes")
    ap.add_argument("--tmp", default="/tmp/clsnorm")
    args = ap.parse_args()

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(d for d in os.listdir(args.result_dirs[0])
                        if os.path.isdir(os.path.join(args.result_dirs[0], d)))

    for rd in args.result_dirs:
        pred_files, recon_pcs, gt_files, gt_pcs, valid = [], [], [], [], []
        total_unmatched = 0
        for seq in scenes:
            npz = os.path.join(rd, seq, "ckpt_final.npz")
            ply = os.path.join(rd, seq, "final.ply")
            gf = os.path.join(args.gt_seg_dir, seq + ".txt")
            gp = os.path.join(args.gt_dir, seq, seq + "_vh_clean_2.ply")
            if not all(os.path.isfile(x) for x in (npz, ply, gf, gp)):
                print(f"[skip] {rd}/{seq}: missing files")
                continue
            out_npz = os.path.join(args.tmp, rd.replace("/", "_"), seq + ".npz")
            total_unmatched += normalize_scene_npz(npz, out_npz)
            pred_files.append(out_npz)
            recon_pcs.append(ply)
            gt_files.append(gf)
            gt_pcs.append(gp)
            valid.append(seq)
        print(f"\n=== {rd}: {len(valid)} scenes, class-aware (unmatched pred labels: {total_unmatched}) ===")
        if not valid:
            continue
        scene_results, failed = ev.evaluate(valid, pred_files, gt_files, recon_pcs, gt_pcs, output_file=None)
        if scene_results:
            n = len(scene_results)
            print(f"average AP {np.mean([r['ap'] for r in scene_results]):.4f} | "
                  f"AP50 {np.mean([r['ap50'] for r in scene_results]):.4f} | "
                  f"AP75 {np.mean([r['ap75'] for r in scene_results]):.4f} | "
                  f"AP25 {np.mean([r['ap25'] for r in scene_results]):.4f} "
                  f"({n} scenes, failed {len(failed)})")


if __name__ == "__main__":
    main()
