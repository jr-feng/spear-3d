#!/usr/bin/env python3
"""
E8 evidence (mechanism a - under-segmentation): per-NYU40-class predicted coverage on
SceneNN. For every GT class, we measure how much of its GT vertices are covered by ANY
of our reconstructed instances (class-agnostic coverage: does SAM3 even segment it?).

Prediction: vocabulary-covered classes (wall/floor/chair/...) have high coverage;
catch-all classes (otherprops/otherfurniture/otherstructure) that the per-scene prompt
vocabulary cannot address have near-zero coverage -> "vocabulary-missing objects are
not segmented at all, hence low vertex counts" (mechanism a).

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/scenenn_seg_coverage.py \
      --pred-root output_test/sceneNN_sam3_test \
      --gt-zip eval/sceneNN/sceneNN_gt_seg.zip \
      --scans-root data_eval/scenenn \
      --out ground_ral/scenenn_seg_coverage.json
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

_real_argv = list(sys.argv)
sys.argv = ["evaluate_seqs_scannet.py", "--result_dir", ".", "--gt_dir", "."]
import evaluate_seqs_scannet as ev  # noqa: E402
from eval.utils_3d import align_gt_to_recon, get_instances_in_GT_pc  # noqa: E402
sys.argv = _real_argv

NYU40 = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa",
    7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
    12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16: "curtain",
    17: "dresser", 18: "pillow", 19: "mirror", 20: "floor mat", 21: "clothes",
    22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "shower curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "night stand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture",
    40: "otherprops",
}
CATCHALL = {38, 39, 40}  # NYU40 catch-all classes without a ScanNet200-concept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=os.path.join(ROOT, "output_test", "sceneNN_sam3_test"))
    ap.add_argument("--gt-zip", default=os.path.join(ROOT, "eval", "sceneNN", "sceneNN_gt_seg.zip"))
    ap.add_argument("--scans-root", default=os.path.join(ROOT, "data_eval", "scenenn"))
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "ground_ral", "scenenn_seg_coverage.json"))
    args = ap.parse_args()

    zip_txt = {}
    with zipfile.ZipFile(args.gt_zip) as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                zip_txt[os.path.basename(n)[:-4]] = n

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(zip_txt)

    # class_id -> list of coverage values (per scene, vertex-weighted)
    cov_tot = defaultdict(lambda: [0, 0])  # cid -> [covered_vert, gt_vert]
    per_scene = []
    for s in scenes:
        npz = os.path.join(args.pred_root, s, "ckpt_final.npz")
        ply = os.path.join(args.pred_root, s, "final.ply")
        gt_mesh = os.path.join(args.scans_root, s, f"{s}.ply")
        if not all(os.path.isfile(x) for x in (npz, ply, gt_mesh)) or s not in zip_txt:
            print(f"[skip] {s}")
            continue
        try:
            gt_ids = np.loadtxt(io.BytesIO(open(args.gt_zip, "rb").read())) if False else None
        except Exception:
            pass
        with zipfile.ZipFile(args.gt_zip) as zf:
            gt_ids = np.loadtxt(io.BytesIO(zf.read(zip_txt[s])), dtype=np.int64)
        gt_class = gt_ids // 1000

        recon_pc = o3d.io.read_point_cloud(ply)
        gt_pc = o3d.io.read_point_cloud(gt_mesh)
        if np.asarray(gt_pc.points).shape[0] != gt_ids.shape[0]:
            print(f"[skip] {s}: GT txt {gt_ids.shape[0]} != mesh {np.asarray(gt_pc.points).shape[0]}")
            continue

        pred_info_raw = ev.read_prediction_npz(npz)
        inst_masks = np.stack([v["mask"] for v in pred_info_raw.values()], axis=0)
        valid_recon = np.any(inst_masks, axis=0)
        corr, valid_idx = align_gt_to_recon(gt_pc, recon_pc, valid_recon_pts_mask=valid_recon,
                                            distance_upper_bound=0.15)
        pred_info = get_instances_in_GT_pc(pred_info_raw, corr, valid_gt_pt_indices=valid_idx)
        covered = np.zeros(gt_ids.shape[0], dtype=bool)
        for inst in pred_info.values():
            m = np.asarray(inst["mask"], dtype=bool)
            if len(m) == covered.shape[0]:
                covered |= m

        scene_row = {"scene": s}
        for cid in np.unique(gt_class):
            cid = int(cid)
            if cid == 0:
                continue
            gt_c = gt_class == cid
            n_gt = int(gt_c.sum())
            if n_gt < 100:
                continue
            n_cov = int((gt_c & covered).sum())
            cov_tot[cid][0] += n_cov
            cov_tot[cid][1] += n_gt
            scene_row[NYU40.get(cid, str(cid))] = round(n_cov / n_gt, 3)
        per_scene.append(scene_row)

    print(f"scenes evaluated: {len(per_scene)}\n")
    print(f"{'NYU40 class':<18}{'GT verts':>12}{'covered':>10}{'cov%':>8}  vocab?")
    rows = []
    for cid in sorted(cov_tot):
        cov, tot = cov_tot[cid]
        name = NYU40.get(cid, f"id{cid}")
        in_vocab = cid not in CATCHALL
        rows.append({"nyu40_id": cid, "name": name, "gt_vertices": tot,
                     "covered": cov, "coverage": cov / max(tot, 1), "catchall": not in_vocab})
        print(f"{name:<18}{tot:>12}{cov:>10}{cov/max(tot,1)*100:>7.1f}%  {'in' if in_vocab else 'CATCHALL'}")

    non_catch = [r for r in rows if not r["catchall"]]
    catch = [r for r in rows if r["catchall"]]
    def wavg(rs):
        t = sum(r["gt_vertices"] for r in rs)
        return sum(r["covered"] for r in rs) / max(t, 1)
    print(f"\n=== vertex-weighted coverage ===")
    print(f"non-catchall classes ({len(non_catch)}): {wavg(non_catch)*100:.1f}%")
    if catch:
        print(f"catch-all classes ({len(catch)}): {wavg(catch)*100:.1f}%  <- under-segmentation evidence")

    out = {"scenes": [p["scene"] for p in per_scene], "rows": rows,
           "coverage_non_catchall": wavg(non_catch), "coverage_catchall": wavg(catch)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
