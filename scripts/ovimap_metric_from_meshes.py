#!/usr/bin/env python3
"""
E7 helper — recompute OVI-MAP's official mIoU / mAcc from ALREADY-BUILT
semantic meshes, without rebuilding anything.

scripts/ovimap_aligned_eval.py always rebuilds gt/pred semantic meshes under
its --tmp dir; when those meshes already exist (e.g. /tmp/ovimap30), this
script reuses them and calls OVI-MAP's own eval_per_class_IoU directly, so the
metric can be re-read in seconds.

Mesh layout expected (written by ovimap_aligned_eval.build_scene_meshes):
    <tmp>/<result_dir with '/' -> '_'>/<scene>/gt_semantic_mesh.ply
    <tmp>/<result_dir with '/' -> '_'>/<scene>/semantic_map_gt_<scene>.ply

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/ovimap_metric_from_meshes.py \
      --tmp /tmp/ovimap30 \
      --tags _home_OnlineAnySeg_e5_out_200_dense \
      --scenes "$(cat e3_scenes_30.txt | tr '\n' ',')"
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "OVI-MAP"))
sys.path.insert(0, os.path.join(ROOT, "OVI-MAP", "scripts"))

from scripts.utils.semantic_const import CLASS_LABELS_200, VALID_CLASS_IDS_200  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True, help="dir holding per-dir/per-scene semantic meshes")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="result-dir tag(s), i.e. result_dir.replace('/', '_')")
    ap.add_argument("--scenes", required=True, help="comma-separated scenes")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    from scripts.eval_sem_seg import eval_per_class_IoU  # OVI-MAP's official metric

    valid_ids = [0] + list(VALID_CLASS_IDS_200)
    labels = ["background"] + list(CLASS_LABELS_200)

    for tag in args.tags:
        cfg_gt, cfg_pred, used = [], [], []
        for seq in scenes:
            d = os.path.join(args.tmp, tag, seq)
            gtf = os.path.join(d, "gt_semantic_mesh.ply")
            pf = os.path.join(d, f"semantic_map_gt_{seq}.ply")
            if not (os.path.isfile(gtf) and os.path.isfile(pf)):
                continue
            cfg_gt.append({"sem_mesh_f": gtf, "inst_mesh_f": gtf, "res_folder": d})
            cfg_pred.append({"sem_mesh_f": pf, "inst_mesh_f": pf, "res_folder": d})
            used.append(seq)
        print(f"\n=== {tag}: {len(used)} scenes (meshes reused from {args.tmp}) ===")
        if not cfg_gt:
            print("  no meshes found — run scripts/ovimap_aligned_eval.py first")
            continue
        eval_per_class_IoU(cfg_gt, cfg_pred, valid_ids, labels)


if __name__ == "__main__":
    main()
