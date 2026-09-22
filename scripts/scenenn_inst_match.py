#!/usr/bin/env python3
"""
E8 evidence (mechanism a - under-segmentation): per-NYU40-class INSTANCE-level matching
recall on SceneNN.

Raw per-vertex "any instance covers this GT vertex" coverage is NOT informative: OAS
instances tile the reconstructed surface, so a bottle sitting on a table is "covered" by
the table mask even when SAM3 never outputs a mask for the bottle. The discriminator for
under-segmentation is instance-level: does a DEDICATED predicted instance overlap each GT
instance at IoU >= tau?

We therefore report, per NYU40 class, the fraction of GT instances that are matched by any
predicted instance at IoU 0.25/0.5/0.75 (class-agnostic matching - does SAM3+OAS segment
this object at all?), restricted to GT instances whose geometry is actually reconstructed
(support filter), so reconstruction failure is not confused with under-segmentation.

Prediction (mechanism a): classes with a ScanNet200 concept in the vocabulary (chair, bed,
sofa, table, ...) have high matching recall; NYU40 catch-all classes that no vocabulary
concept can address (otherprops/otherfurniture/otherstructure) have near-zero matching
recall - SAM3 outputs no masks for those objects, so no predicted instance is dedicated to
them. This quantifies that the SceneNN deficit is a vocabulary/annotation mismatch
(design consequence), not a failure of segmentation on vocabulary-covered classes.

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/scenenn_inst_match.py \
      --pred-root output_test/sceneNN_sam3_test \
      --gt-zip eval/sceneNN/sceneNN_gt_seg.zip \
      --scans-root data_eval/scenenn \
      --vocab-json ground_ral/scenenn_vocab_coverage.json \
      --out ground_ral/scenenn_inst_match.json
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
from eval.utils_3d import align_gt_to_recon  # noqa: E402
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

IouT = (0.25, 0.5, 0.75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=os.path.join(ROOT, "output_test", "sceneNN_sam3_test"))
    ap.add_argument("--gt-zip", default=os.path.join(ROOT, "eval", "sceneNN", "sceneNN_gt_seg.zip"))
    ap.add_argument("--scans-root", default=os.path.join(ROOT, "data_eval", "scenenn"))
    ap.add_argument("--vocab-json", default=os.path.join(ROOT, "ground_ral", "scenenn_vocab_coverage.json"))
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--dist", type=float, default=0.15)
    ap.add_argument("--min-support-verts", type=int, default=50,
                    help="GT instance needs >= this many reconstructed (valid) vertices")
    ap.add_argument("--min-support-ratio", type=float, default=0.3,
                    help="GT instance needs >= this fraction of vertices reconstructed")
    ap.add_argument("--out", default=os.path.join(ROOT, "ground_ral", "scenenn_inst_match.json"))
    args = ap.parse_args()

    vocab = json.load(open(args.vocab_json, encoding="utf-8"))
    in_vocab = {r["nyu40_id"]: r["in_scannet200_vocab"] for r in vocab["rows"]}

    zip_txt = {}
    with zipfile.ZipFile(args.gt_zip) as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                zip_txt[os.path.basename(n)[:-4]] = n

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(zip_txt)

    # cid -> per-threshold counters over GT instances
    cnt = defaultdict(lambda: {"tot": 0, "match": {t: 0 for t in IouT},
                               "iou_sum": 0.0, "best_cls": defaultdict(int),
                               "excl": 0, "excl_small": 0})
    per_scene = []
    for s in scenes:
        npz = os.path.join(args.pred_root, s, "ckpt_final.npz")
        ply = os.path.join(args.pred_root, s, "final.ply")
        gt_mesh = os.path.join(args.scans_root, s, f"{s}.ply")
        if not all(os.path.isfile(x) for x in (npz, ply, gt_mesh)) or s not in zip_txt:
            print(f"[skip] {s}")
            continue
        with zipfile.ZipFile(args.gt_zip) as zf:
            gt_ids = np.loadtxt(io.BytesIO(zf.read(zip_txt[s])), dtype=np.int64)

        recon_pc = o3d.io.read_point_cloud(ply)
        gt_pc = o3d.io.read_point_cloud(gt_mesh)
        if np.asarray(gt_pc.points).shape[0] != gt_ids.shape[0]:
            print(f"[skip] {s}: GT txt {gt_ids.shape[0]} != mesh {np.asarray(gt_pc.points).shape[0]}")
            continue

        z = np.load(npz, allow_pickle=False)
        pred_masks = z["pred_masks"]  # (n_recon_pts, n_inst) bool
        n_inst = pred_masks.shape[1]
        n_recon = pred_masks.shape[0]
        if n_recon != np.asarray(recon_pc.points).shape[0]:
            print(f"[skip] {s}: npz pts {n_recon} != ply pts {np.asarray(recon_pc.points).shape[0]}")
            continue

        # ---- alignment: GT pc -> recon pc (returns recon-space mask index per GT vert) ----
        valid_recon = np.any(pred_masks, axis=1)
        corr, valid_gt_idx = align_gt_to_recon(gt_pc, recon_pc, valid_recon_pts_mask=valid_recon,
                                               distance_upper_bound=args.dist)
        # corr[gt_vert] = recon index or -1; valid_gt_idx = GT verts with a recon neighbour

        # ---- pred instance ownership on recon pts (masks are disjoint) ----
        recon_owner = np.full(n_recon, -1, dtype=np.int64)
        for i in range(n_inst):
            m = pred_masks[:, i]
            recon_owner[m] = i
        # map onto GT-vertex space
        gt_owner = np.full(gt_ids.shape[0], -1, dtype=np.int64)
        gt_owner[valid_gt_idx] = recon_owner[corr[valid_gt_idx]]

        # pred instance sizes on valid GT verts (for IoU denominator)
        sizes_p = np.bincount(gt_owner[gt_owner >= 0], minlength=n_inst + 1)[:n_inst]

        # pred instance class names (vocab words) for diagnostics
        pred_cls_names = np.asarray(z["pred_classes"]).astype(str)

        # ---- per-scene per-class counters for detail view ----
        sc_cnt = defaultdict(lambda: {"tot": 0, "match": {t: 0 for t in IouT},
                                      "excl": 0, "excl_small": 0})

        # ---- GT instances: (class, inst) -> vertices ----
        cls = gt_ids // 1000
        inst = gt_ids % 1000
        gtv = np.arange(gt_ids.shape[0])
        n_gt_inst = 0
        n_eval = 0
        for c in np.unique(cls):
            c = int(c)
            if c not in NYU40:
                continue  # aggregate only the 40 NYU40 classes
            for i in np.unique(inst[cls == c]):
                i = int(i)
                if i == 0:
                    continue
                n_gt_inst += 1
                v = gtv[(cls == c) & (inst == i)]
                # only verts that have recon support count toward matching
                valid_v = v[gt_owner[v] >= 0]
                if valid_v.size < args.min_support_verts:
                    cnt[c]["excl_small"] += 1
                    sc_cnt[c]["excl_small"] += 1
                    continue
                if valid_v.size / max(v.size, 1) < args.min_support_ratio:
                    cnt[c]["excl"] += 1
                    sc_cnt[c]["excl"] += 1
                    continue
                n_eval += 1
                owners = gt_owner[valid_v]
                inter = np.bincount(owners, minlength=n_inst)[:n_inst]  # |g ∩ p| per pred p
                denom = valid_v.size + sizes_p - inter
                with np.errstate(divide="ignore", invalid="ignore"):
                    ious = np.where(denom > 0, inter / np.maximum(denom, 1), 0.0)
                best = float(ious.max()) if n_inst else 0.0
                best_p = int(ious.argmax()) if n_inst and ious.max() > 0 else -1
                cnt[c]["tot"] += 1
                cnt[c]["iou_sum"] += best
                sc_cnt[c]["tot"] += 1
                for t in IouT:
                    if best >= t:
                        cnt[c]["match"][t] += 1
                        sc_cnt[c]["match"][t] += 1
                if best_p >= 0:
                    cnt[c]["best_cls"][pred_cls_names[best_p]] += 1
        per_scene.append({"scene": s, "gt_instances": n_gt_inst, "evaluated": n_eval,
                          "pred_instances": n_inst,
                          "valid_gt_verts": int(valid_gt_idx.size),
                          "per_class": {str(c): {"name": NYU40[c],
                                                  "tot": sc_cnt[c]["tot"],
                                                  "recall": {str(t): sc_cnt[c]["match"][t] / max(sc_cnt[c]["tot"], 1)
                                                             for t in IouT},
                                                  "matched": {str(t): sc_cnt[c]["match"][t] for t in IouT},
                                                  "excl": sc_cnt[c]["excl"] + sc_cnt[c]["excl_small"]}
                                        for c in sc_cnt}})

    print(f"scenes evaluated: {len(per_scene)} | "
          f"GT instances seen: {sum(p['gt_instances'] for p in per_scene)}\n")
    print(f"{'NYU40 class':<18}{'inst':>6}{'R@25':>7}{'R@50':>7}{'R@75':>7}{'meanIoU':>9}  vocab")
    rows = []
    for cid in sorted(cnt):
        name = NYU40.get(cid, f"id{cid}")
        tot = cnt[cid]["tot"]
        r = {"nyu40_id": cid, "nyu40_name": name, "instances": tot,
             "recall": {str(t): cnt[cid]["match"][t] / max(tot, 1) for t in IouT},
             "matched": {str(t): cnt[cid]["match"][t] for t in IouT},
             "mean_best_iou": cnt[cid]["iou_sum"] / max(tot, 1),
             "excluded_no_support": cnt[cid]["excl"] + cnt[cid]["excl_small"],
             "excluded_small_under50": cnt[cid]["excl_small"],
             "best_matching_pred_class": dict(cnt[cid]["best_cls"]),
             "in_scannet200_vocab": bool(in_vocab.get(cid, True))}
        rows.append(r)
        print(f"{name:<18}{tot:>6}"
              f"{r['recall']['0.25']*100:>6.1f}%{r['recall']['0.5']*100:>6.1f}%"
              f"{r['recall']['0.75']*100:>6.1f}%{r['mean_best_iou']:>9.3f}"
              f"  {'in' if r['in_scannet200_vocab'] else 'OUT'}")

    def wrec(rs, t):
        tot = sum(r["instances"] for r in rs)
        mat = sum(r["matched"][str(t)] for r in rs)
        return mat / max(tot, 1)

    in_r = [r for r in rows if r["in_scannet200_vocab"]]
    out_r = [r for r in rows if not r["in_scannet200_vocab"]]
    print(f"\n=== instance-level matching recall (GT instances w/ reconstructed support) ===")
    for t in IouT:
        print(f"  IoU>={t}: vocab-in classes ({sum(r['instances'] for r in in_r)} inst): "
              f"{wrec(in_r, t)*100:.1f}% | vocab-OUT classes ({sum(r['instances'] for r in out_r)} inst): "
              f"{wrec(out_r, t)*100:.1f}%")
    print(f"\n  catch-all only (otherprops/otherfurniture/otherstructure):")
    ca = [r for r in rows if r["nyu40_id"] in (38, 39, 40)]
    for t in IouT:
        print(f"    IoU>={t}: {wrec(ca, t)*100:.1f}%  ({sum(r['instances'] for r in ca)} inst)")

    out = {"scenes": per_scene, "rows": rows,
           "recall_vocab_in": {str(t): wrec(in_r, t) for t in IouT},
           "recall_vocab_out": {str(t): wrec(out_r, t) for t in IouT},
           "recall_catchall": {str(t): wrec(ca, t) for t in IouT},
           "n_inst_vocab_in": sum(r["instances"] for r in in_r),
           "n_inst_vocab_out": sum(r["instances"] for r in out_r),
           "n_inst_catchall": sum(r["instances"] for r in ca),
           "params": {"dist": args.dist, "min_support_verts": args.min_support_verts,
                      "min_support_ratio": args.min_support_ratio}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
