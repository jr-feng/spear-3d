#!/usr/bin/env python3
"""
E8 mechanism-a: per-NYU40-class CLASS-MATCHED vertex coverage on SceneNN.
Covers the metric actually reported for SceneNN per-class evaluation:
GT vertices of class c are "covered" only when they lie inside a predicted
instance whose assigned class name maps back to NYU40 class c (via the same
NYU40<->ScanNet200 alias table used for vocabulary coverage). For NYU40
catch-all classes (otherprops/otherfurniture/otherstructure) no ScanNet200
concept exists, so class-matched coverage is ~0 BY CONSTRUCTION -> low
per-class vertex counts -> the SceneNN gap is a vocabulary/annotation
mismatch (design consequence), on vocabulary-covered classes the method
performs well.

Also reports GT annotation void share (class-0 / unlabelled vertices) to
quantify the "incomplete annotation" component quantitatively.

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/scenenn_class_coverage.py \
      --pred-root output_test/sceneNN_sam3_test \
      --gt-zip eval/sceneNN/sceneNN_gt_seg.zip \
      --scans-root data_eval/scenenn \
      --vocab-json ground_ral/scenenn_vocab_coverage.json \
      --out ground_ral/scenenn_class_coverage.json
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

# pred vocab word (normalized) -> NYU40 id, built from the same conceptual
# mapping used in scenenn_vocab_coverage.py (exact ScanNet200 names + aliases).
def _norm(s):
    return " ".join(str(s or "").strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=os.path.join(ROOT, "output_test", "sceneNN_sam3_test"))
    ap.add_argument("--gt-zip", default=os.path.join(ROOT, "eval", "sceneNN", "sceneNN_gt_seg.zip"))
    ap.add_argument("--scans-root", default=os.path.join(ROOT, "data_eval", "scenenn"))
    ap.add_argument("--vocab-json", default=os.path.join(ROOT, "ground_ral", "scenenn_vocab_coverage.json"))
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--dist", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(ROOT, "ground_ral", "scenenn_class_coverage.json"))
    args = ap.parse_args()

    vocab = json.load(open(args.vocab_json, encoding="utf-8"))
    in_vocab = {r["nyu40_id"]: r["in_scannet200_vocab"] for r in vocab["rows"]}

    # word -> NYU40 name mapping (reverse of the coverage concept table)
    word_to_cid = {}
    for cid, name in NYU40.items():
        if not in_vocab.get(cid, True):
            continue
        cands = {name}  # ScanNet200 synonyms; never shadow an exact NYU40 name
        if name == "sofa":
            cands.add("couch")
        if name == "shelves":
            cands.add("shelf")
        if name == "books":
            cands.add("book")
        if name == "night stand":
            cands.add("nightstand")
        if name == "blinds":
            cands.add("window blind")
        for w in sorted(cands):
            wn = _norm(w)
            if wn not in word_to_cid:  # keep first (exact NYU40 name) occurrence
                word_to_cid[wn] = cid

    zip_txt = {}
    with zipfile.ZipFile(args.gt_zip) as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                zip_txt[os.path.basename(n)[:-4]] = n

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(zip_txt)

    stat = defaultdict(lambda: {"gt_vert": 0, "cov_vert": 0, "cov_any_vert": 0, "gt_inst": 0})
    void_stat = defaultdict(lambda: {"tot": 0, "void": 0})
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
        pred_masks = z["pred_masks"]
        n_inst = pred_masks.shape[1]
        n_recon = pred_masks.shape[0]
        if n_recon != np.asarray(recon_pc.points).shape[0]:
            print(f"[skip] {s}: npz pts {n_recon} != ply pts {np.asarray(recon_pc.points).shape[0]}")
            continue

        valid_recon = np.any(pred_masks, axis=1)
        corr, valid_gt_idx = align_gt_to_recon(gt_pc, recon_pc, valid_recon_pts_mask=valid_recon,
                                               distance_upper_bound=args.dist)
        recon_owner = np.full(n_recon, -1, dtype=np.int64)
        for i in range(n_inst):
            recon_owner[pred_masks[:, i]] = i
        gt_owner = np.full(gt_ids.shape[0], -1, dtype=np.int64)
        gt_owner[valid_gt_idx] = recon_owner[corr[valid_gt_idx]]

        pred_cls_names = np.asarray(z["pred_classes"]).astype(str)
        pred_cid = np.array([word_to_cid.get(_norm(w), -1) for w in pred_cls_names])  # NYU40 id or -1

        cls = gt_ids // 1000
        inst = gt_ids % 1000
        n_gt_inst = 0
        for c in np.unique(cls):
            c = int(c)
            if c not in NYU40:
                continue
            m_c = cls == c
            stat[c]["gt_vert"] += int(m_c.sum())
            # coverage by any instance (class-agnostic)
            stat[c]["cov_any_vert"] += int((m_c & (gt_owner >= 0)).sum())
            # class-matched coverage: covered by a pred instance whose name maps to c
            owner_c = gt_owner[m_c]
            ok = np.full(owner_c.shape[0], False)
            valid = owner_c >= 0
            ok[valid] = pred_cid[owner_c[valid]] == c
            stat[c]["cov_vert"] += int(ok.sum())
            for i in np.unique(inst[m_c]):
                i = int(i)
                if i == 0:
                    continue
                n_gt_inst += 1
                stat[c]["gt_inst"] += 1
        void_stat[s]["tot"] = gt_ids.shape[0]
        void_stat[s]["void"] = int((cls == 0).sum())
        per_scene.append({"scene": s, "gt_instances": n_gt_inst, "void_share": void_stat[s]["void"] / max(void_stat[s]["tot"], 1)})

    print(f"scenes: {len(per_scene)}\n")
    print(f"{'NYU40 class':<16}{'gt_v':>10}{'any%':>7}{'clsmatch%':>10}{'gt_inst':>8}  vocab")
    rows = []
    for cid in sorted(stat):
        name = NYU40[cid]
        gt_v = stat[cid]["gt_vert"]
        any_c = stat[cid]["cov_any_vert"] / max(gt_v, 1)
        cm = stat[cid]["cov_vert"] / max(gt_v, 1)
        rows.append({"nyu40_id": cid, "nyu40_name": name,
                     "gt_vertices": gt_v, "gt_instances": stat[cid]["gt_inst"],
                     "coverage_any": any_c, "coverage_class_matched": cm,
                     "in_scannet200_vocab": bool(in_vocab.get(cid, True))})
        print(f"{name:<16}{gt_v:>10}{any_c*100:>6.1f}%{cm*100:>9.1f}%{stat[cid]['gt_inst']:>8}  {'in' if rows[-1]['in_scannet200_vocab'] else 'OUT'}")

    in_r = [r for r in rows if r["in_scannet200_vocab"]]
    out_r = [r for r in rows if not r["in_scannet200_vocab"]]

    def wcov(rs, key):
        t = sum(r["gt_vertices"] for r in rs)
        s = sum(r[key] * r["gt_vertices"] for r in rs)
        return s / max(t, 1)

    print(f"\n=== vertex-weighted coverage ===")
    print(f"vocab-in classes:  any={wcov(in_r,'coverage_any')*100:.1f}%  class-matched={wcov(in_r,'coverage_class_matched')*100:.1f}%  (gt verts {sum(r['gt_vertices'] for r in in_r)})")
    print(f"vocab-OUT classes: any={wcov(out_r,'coverage_any')*100:.1f}%  class-matched={wcov(out_r,'coverage_class_matched')*100:.1f}%  (gt verts {sum(r['gt_vertices'] for r in out_r)})")
    catch = [r for r in rows if r["nyu40_id"] in (38, 39, 40)]
    print(f"catch-all (38/39/40): any={wcov(catch,'coverage_any')*100:.1f}%  class-matched={wcov(catch,'coverage_class_matched')*100:.1f}%  (gt verts {sum(r['gt_vertices'] for r in catch)})")

    void_share = np.mean([p["void_share"] for p in per_scene])
    print(f"\nGT annotation void (class-0 verts): mean per scene = {void_share*100:.1f}%  "
          f"(scenes: {[round(p['void_share']*100,1) for p in per_scene]})")

    out = {"scenes": per_scene, "rows": rows,
           "coverage_any_vocab_in": wcov(in_r, "coverage_any"),
           "coverage_cls_vocab_in": wcov(in_r, "coverage_class_matched"),
           "coverage_any_vocab_out": wcov(out_r, "coverage_any"),
           "coverage_cls_vocab_out": wcov(out_r, "coverage_class_matched"),
           "coverage_cls_catchall": wcov(catch, "coverage_class_matched"),
           "mean_gt_void_share": float(void_share)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
