#!/usr/bin/env python3
"""
E8 Step 1: SceneNN vocabulary-coverage statistics (NYU40 GT vs our ScanNet200 vocabulary).

Reads SceneNN GT segmentations (eval/sceneNN/sceneNN_gt_seg.zip, label = NYU40_id*1000+inst)
and reports, per NYU40 class: instance count / vertex share, whether a conceptual
correspondence exists in our ScanNet200 vocabulary, and the theoretical upper bound
of correctly classifiable GT instances given the vocabulary (R1: "SceneNN loss is a
consequence of the vocabulary design").

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/scenenn_vocab_coverage.py \
      --gt-zip /home/OnlineAnySeg/eval/sceneNN/sceneNN_gt_seg.zip \
      --out ground_ral/scenenn_vocab_coverage.json
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "scannet200"))
from scannet200_constants import CLASS_LABELS_200  # noqa: E402

# ---------------- NYU40 class table (id -> name) ----------------
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

SCANNET200 = [str(x).strip().lower() for x in CLASS_LABELS_200]
SCANNET200_SET = set(SCANNET200)

# manual conceptual correspondences NYU40 -> ScanNet200 (beyond exact name match)
NYU40_TO_SCANNET = {
    "television": "television", "night stand": "nightstand",
    "floor mat": "floor", "otherstructure": None, "otherfurniture": None,
    "otherprops": None, "clothes": "clothes", "shelves": "shelf",
    "picture": "picture", "blinds": "window blind", "curtain": "curtain",
    "sofa": "couch",
    "dresser": "dresser", "books": "book", "paper": "paper",
    "shower curtain": "shower curtain", "whiteboard": "whiteboard",
    "towel": "towel", "counter": "counter", "desk": "desk",
    "pillow": "pillow", "mirror": "mirror", "refrigerator": "refrigerator",
    "bathtub": "bathtub", "bag": None, "box": None, "lamp": "lamp",
    "sink": "sink", "toilet": "toilet", "person": "person",
    "ceiling": "ceiling", "bookshelf": "bookshelf", "curtain": "curtain",
}


def _norm(s):
    return " ".join(str(s or "").strip().lower().split())


def _has_scannet_concept(nyu_name: str) -> bool:
    n = _norm(nyu_name)
    if n in SCANNET200_SET:
        return True
    mapped = NYU40_TO_SCANNET.get(n)
    if mapped is None:
        return False
    return _norm(mapped) in SCANNET200_SET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-zip", default=os.path.join(ROOT, "eval", "sceneNN", "sceneNN_gt_seg.zip"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ground_ral", "scenenn_vocab_coverage.json"))
    args = ap.parse_args()

    # ---- aggregate per-class instances & vertices over all scenes in the zip ----
    class_inst: dict = defaultdict(set)   # class_id -> set of (scene, inst)
    class_vert: dict = defaultdict(int)   # class_id -> vertex count
    n_scenes = 0
    with zipfile.ZipFile(args.gt_zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".txt"):
                continue
            scene = os.path.basename(name)[:-4]
            n_scenes += 1
            raw = np.loadtxt(io.BytesIO(zf.read(name)), dtype=np.int64)
            ids = raw // 1000
            insts = raw % 1000
            for cid in np.unique(ids):
                cid = int(cid)
                if cid == 0:
                    continue
                m = ids == cid
                class_vert[cid] += int(m.sum())
                class_inst[cid].update((scene, int(i)) for i in insts[m])

    total_inst = sum(len(v) for v in class_inst.values())
    total_vert = sum(class_vert.values())

    rows = []
    for cid in sorted(class_inst):
        name = NYU40.get(cid, f"id{cid}")
        n_inst = len(class_inst[cid])
        n_vert = class_vert[cid]
        has = _has_scannet_concept(name)
        rows.append({
            "nyu40_id": cid, "nyu40_name": name,
            "instances": n_inst, "inst_share": round(n_inst / max(total_inst, 1), 4),
            "vertices": n_vert, "vert_share": round(n_vert / max(total_vert, 1), 4),
            "in_scannet200_vocab": has,
        })

    cover_inst = sum(r["instances"] for r in rows if r["in_scannet200_vocab"])
    cover_vert = sum(r["vertices"] for r in rows if r["in_scannet200_vocab"])
    missing = [r for r in rows if not r["in_scannet200_vocab"]]

    print(f"SceneNN GT scenes in zip: {n_scenes}")
    print(f"NYU40 classes present: {len(rows)} | total instances {total_inst} | vertices {total_vert}\n")
    print(f"{'id':<4}{'NYU40 class':<18}{'inst':>6}{'inst%':>8}{'vert%':>8}  in vocab")
    for r in rows:
        print(f"{r['nyu40_id']:<4}{r['nyu40_name']:<18}{r['instances']:>6}"
              f"{r['inst_share']*100:>7.1f}%{r['vert_share']*100:>7.1f}%  {r['in_scannet200_vocab']}")
    print(f"\n=== coverage (classes with a ScanNet200 concept) ===")
    print(f"instances inside vocab: {cover_inst}/{total_inst} = {cover_inst/total_inst:.3f}")
    print(f"vertices  inside vocab: {cover_vert}/{total_vert} = {cover_vert/total_vert:.3f}")
    print(f"missing classes: {[r['nyu40_name'] for r in missing]}")
    if missing:
        m_inst = sum(r["instances"] for r in missing)
        print(f"missing-class instance share: {m_inst}/{total_inst} = {m_inst/total_inst:.3f}")

    out = {
        "n_scenes": n_scenes,
        "rows": rows,
        "total_instances": total_inst,
        "total_vertices": total_vert,
        "coverage_instances": cover_inst / max(total_inst, 1),
        "coverage_vertices": cover_vert / max(total_vert, 1),
        "missing_classes": [r["nyu40_name"] for r in missing],
        "missing_instances_share": sum(r["instances"] for r in missing) / max(total_inst, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
