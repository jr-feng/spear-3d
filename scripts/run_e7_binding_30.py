#!/usr/bin/env python3
"""Paired label-bound vs post-hoc scores on the same 30 reconstructions.

Official OVI-MAP reconstruction is not runnable in this checkout: it needs a
built ROS workspace and the ovimap-perception environment, and neither is
present. This driver runs scripts/eval_posthoc_sem.py twice on one saved
reconstruction directory and OVI-MAP's eval_per_class_IoU:

  label-bound  stored class name; 0 post-hoc queries per instance
  post-hoc     one CLIP argmax per instance against the 200 ScanNet200 names

The query column is that count. It is not OVI-MAP's published view-query rate.

Usage:
  NS_CLIP_DEVICE=cuda:0 /root/anaconda3/envs/OASeg/bin/python scripts/run_e7_binding_30.py \
      --scenes-file e3_scenes_30.txt --pred-root output_ral/B \
      --out ground_ral/e7_binding_30.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_arm(arm, scenes, pred_root, gt_dir, gt_seg_dir, tmp):
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "eval_posthoc_sem.py"),
        "--arm", arm,
        "--pred-root", pred_root,
        "--gt_dir", gt_dir,
        "--gt_seg_dir", gt_seg_dir,
        "--scenes", ",".join(scenes),
        "--tmp", tmp,
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    text = "".join(lines)
    if rc != 0:
        raise RuntimeError(f"arm {arm} failed ({rc}):\n{text[-4000:]}")
    scene_m = re.search(
        r"(\d+) scenes \| post-hoc queries=(\d+) \| unmatched instances=(\d+) \| "
        r"name_agree=([0-9.]+) \| mean_max_sim=([0-9.]+)",
        text,
    )
    metric = re.findall(r"\n([0-9]+\.[0-9]+)\t([0-9]+\.[0-9]+)\n", text)
    if not scene_m or not metric:
        raise RuntimeError(f"could not parse arm {arm} output:\n{text[-4000:]}")
    miou, macc = metric[-1]
    return {
        "arm": "label-bound" if arm == "A" else "post-hoc",
        "scenes": int(scene_m.group(1)),
        "assignments": int(scene_m.group(2)),
        "unmatched_instances": int(scene_m.group(3)),
        "name_agree": float(scene_m.group(4)),
        "mean_max_sim": float(scene_m.group(5)),
        "mIoU": float(miou),
        "mAcc": float(macc),
        "log_tail": text[-1500:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-file", default=os.path.join(ROOT, "e3_scenes_30.txt"))
    ap.add_argument("--pred-root", default=os.path.join(ROOT, "output_ral", "B"),
                    help="crop-feature reconstructions; e5_out_200_dense stores text embeddings of the class name")
    ap.add_argument("--gt-dir", default=os.path.join(ROOT, "data_eval", "scannet"))
    ap.add_argument("--gt-seg-dir", default=os.path.join(ROOT, "eval", "scannet200", "validation"))
    ap.add_argument("--tmp", default="/tmp/e7_binding_30")
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "ground_ral", "e7_binding_30.json"))
    args = ap.parse_args()

    scenes = [ln.strip() for ln in open(args.scenes_file) if ln.strip()]
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    pred_root = args.pred_root if os.path.isabs(args.pred_root) else os.path.join(ROOT, args.pred_root)

    n_inst = 0
    for seq in scenes:
        npz = os.path.join(pred_root, seq, "ckpt_final.npz")
        if os.path.isfile(npz):
            z = np.load(npz, allow_pickle=False)
            n_inst += int(np.asarray(z["pred_classes"]).shape[0])

    rows = []
    for arm in ("A", "B"):
        row = _run_arm(arm, scenes, pred_root, args.gt_dir, args.gt_seg_dir, args.tmp)
        row["instances"] = n_inst
        row["queries_per_instance"] = 0.0 if arm == "A" else (1.0 if n_inst else 0.0)
        rows.append(row)
        print(
            f"{row['arm']}: scenes={row['scenes']} mIoU={row['mIoU']:.4f} "
            f"mAcc={row['mAcc']:.4f} queries/instance={row['queries_per_instance']} "
            f"name_agree={row['name_agree']:.3f} mean_max_sim={row['mean_max_sim']:.3f}",
            flush=True,
        )

    payload = {
        "pred_root": pred_root,
        "n_scenes_listed": len(scenes),
        "instances": n_inst,
        "metric": "OVI-MAP eval_per_class_IoU",
        "protocol": (
            "Same output_ral/B reconstructions (full ScanNet sequence, SAM3 masks, "
            "per-crop CLIP features). Label-bound uses the stored SAM3 class name "
            "(0 post-hoc queries per instance). Post-hoc is one argmax of that crop "
            "feature against the 200 ScanNet200 CLIP text embeddings. This is not an "
            "OVI-MAP reconstruction and not OVI-MAP's published view-query count."
        ),
        "rows": [{k: v for k, v in r.items() if k != "log_tail"} for r in rows],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
