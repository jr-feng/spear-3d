#!/usr/bin/env python3
"""Target-instance accuracy on saved grounding logs.

A query is correct only when the selected reconstructed instance's
highest point-IoU on the ScanNet GT mesh is the referred object, and
that IoU is at least the threshold. A nearby object with a close centroid
does not count. Queries with no selected instance count as incorrect.

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/target_instance_acc.py \
      --pred-root output_test/nr3d \
      --logs qwen=output_test/nr3d/qwen_outputs_y.json \
      --out ground_ral/target_instance_nr3d.json
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_records(path):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            return data if isinstance(data, list) else []
        return [json.loads(line) for line in f if line.strip()]


def _scene_iou(scene_id, pred_root, scan_root):
    info_path = os.path.join(pred_root, scene_id, "pred_instance_info.npz")
    seg_path = os.path.join(scan_root, scene_id, scene_id + "_vh_clean_2.0.010000.segs.json")
    agg_path = os.path.join(scan_root, scene_id, scene_id + "_vh_clean.aggregation.json")
    if not all(os.path.isfile(p) for p in (info_path, seg_path, agg_path)):
        return None
    info = np.load(info_path, allow_pickle=True)
    masks = np.asarray(info["pred_instance_masks"], dtype=bool)
    segs = json.load(open(seg_path))
    seg_idx = np.asarray(segs["segIndices"], dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != seg_idx.shape[0]:
        return None
    agg = json.load(open(agg_path))
    seg_to_obj = {}
    for group in agg["segGroups"]:
        oid = int(group["objectId"])
        for seg in group["segments"]:
            seg_to_obj[int(seg)] = oid
    vert_obj = np.fromiter((seg_to_obj.get(int(s), -1) for s in seg_idx), dtype=np.int32, count=seg_idx.shape[0])
    obj_ids = [int(x) for x in np.unique(vert_obj) if int(x) >= 0]
    if not obj_ids:
        return None
    gt = np.stack([vert_obj == oid for oid in obj_ids], axis=0)
    pred_i = masks.astype(np.int32)
    gt_i = gt.astype(np.int32)
    inter = pred_i @ gt_i.T
    pred_sum = pred_i.sum(axis=1, keepdims=True)
    gt_sum = gt_i.sum(axis=1)
    union = pred_sum + gt_sum.reshape(1, -1) - inter
    iou = inter / np.maximum(union, 1)
    return {"obj_ids": np.asarray(obj_ids, dtype=np.int32), "iou": iou.astype(np.float32)}


def _score_log(name, records, pred_root, scan_root, cache):
    n = len(records)
    no_pred = 0
    parse_fail = 0
    missing_scene = 0
    centroid_hit = 0
    centroid_known = 0
    hit25 = 0
    hit50 = 0
    centroid_but_other = 0
    for rec in records:
        if rec.get("parse_error"):
            parse_fail += 1
        scene_id = rec.get("scene_id")
        if scene_id not in cache:
            cache[scene_id] = _scene_iou(scene_id, pred_root, scan_root)
        scene = cache[scene_id]
        if scene is None:
            missing_scene += 1
            continue
        pred_id = rec.get("best_pred_id")
        dist = rec.get("distance")
        if dist is not None:
            centroid_known += 1
        centroid_ok = dist is not None and float(dist) <= 0.5
        if centroid_ok:
            centroid_hit += 1
        if pred_id is None:
            no_pred += 1
            continue
        pred_id = int(pred_id)
        if pred_id < 0 or pred_id >= scene["iou"].shape[0]:
            continue
        row = scene["iou"][pred_id]
        target = int(rec.get("object_id"))
        target_pos = np.where(scene["obj_ids"] == target)[0]
        target_iou = float(row[target_pos[0]]) if len(target_pos) else 0.0
        best = float(row.max()) if row.size else 0.0
        others = row.copy()
        if len(target_pos):
            others[target_pos[0]] = -1.0
        best_other = float(others.max()) if others.size else -1.0
        unique_target = target_iou > best_other
        if unique_target and target_iou >= 0.25:
            hit25 += 1
        if unique_target and target_iou >= 0.50:
            hit50 += 1
        if centroid_ok and not (unique_target and target_iou >= 0.25):
            if best_other > target_iou:
                centroid_but_other += 1
    scored = n - missing_scene
    def pct(k, d):
        return round(100.0 * k / d, 2) if d else None
    return {
        "name": name,
        "queries": n,
        "scored": scored,
        "missing_scene": missing_scene,
        "no_prediction": no_pred,
        "parse_fail": parse_fail,
        "centroid_known": centroid_known,
        "acc05_centroid_all": pct(centroid_hit, scored),
        "acc05_centroid_known": pct(centroid_hit, centroid_known),
        "target_acc_iou25": pct(hit25, scored),
        "target_acc_iou50": pct(hit50, scored),
        "centroid_hit_other_object": centroid_but_other,
        "centroid_hit_other_object_pct": pct(centroid_but_other, scored),
    }


def _csvg_to_records(path, id_map):
    raw = json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path)))
    records = []
    for rec in raw:
        scene_id = rec.get("scene_id")
        sel = rec.get("selected_mask3d_id")
        pred_id = None
        if sel is not None:
            mapped = (id_map.get(scene_id) or {}).get(str(sel))
            pred_id = int(mapped) if mapped is not None else None
        records.append({
            "scene_id": scene_id,
            "object_id": rec.get("target_id"),
            "best_pred_id": pred_id,
            "distance": None,
            "parse_error": None,
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--scan-root", default=os.path.join(ROOT, "data_eval", "scannet"))
    ap.add_argument("--logs", default="", help="name=path,name=path")
    ap.add_argument("--csvg", default="", help="name=path,name=path of CSVG eval_results json")
    ap.add_argument("--id-map", default=os.path.join(ROOT, "CSVG", "data", "eval_output", "mask3d_val", "pred_id_map.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pred_root = args.pred_root if os.path.isabs(args.pred_root) else os.path.join(ROOT, args.pred_root)
    cache = {}
    rows = []
    jobs = []
    if args.logs:
        for item in args.logs.split(","):
            name, path = item.split("=", 1)
            jobs.append((name, _load_records(path)))
    if args.csvg:
        id_map = json.load(open(args.id_map))
        for item in args.csvg.split(","):
            name, path = item.split("=", 1)
            jobs.append((name, _csvg_to_records(path, id_map)))
    if not jobs:
        raise SystemExit("need --logs or --csvg")
    for name, records in jobs:
        row = _score_log(name, records, pred_root, args.scan_root, cache)
        rows.append(row)
        print(
            f"{name}: n={row['queries']} scored={row['scored']} "
            f"no_pred={row['no_prediction']} parse_fail={row['parse_fail']} "
            f"Acc@0.5m={row['acc05_centroid_all']} "
            f"T-Acc@0.25={row['target_acc_iou25']} T-Acc@0.50={row['target_acc_iou50']} "
            f"centroid-but-other={row['centroid_hit_other_object']}",
            flush=True,
        )
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "pred_root": pred_root,
        "definition": (
            "Correct iff the referred ScanNet object is the unique highest "
            "point-IoU match of the selected reconstructed instance, and IoU "
            "is at least the threshold. No selection counts incorrect. "
            "Denominator is queries whose scene masks and aggregation loaded."
        ),
        "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
