#!/usr/bin/env python3
"""Replay the saved scorer-ablation queries and score T-Acc.

Uses the same grounder settings as ablate_scorers.py. T-Acc follows
scripts/target_instance_acc.py: a query is correct when the referred
ScanNet object is the unique highest point-IoU of the selected instance
and that IoU meets the threshold. No selection counts incorrect.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from LLM_sam3_query_graph_clip import (  # noqa: E402
    CLIPTextMatcher,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    QueryGraph,
    QueryGraphGrounder,
    SCORER_NAMES,
    _compute_distance_for_prediction,
    _corresponds,
    _summarize_dists,
)
from spatial_scene_graph import SpatialSceneGraph  # noqa: E402
from scripts.target_instance_acc import _scene_iou  # noqa: E402

BALANCED = os.path.join(ROOT, "ground_ral/nr3d/ablation/ns_ablation_balanced.jsonl")
GT = os.path.join(ROOT, "nr3d/nr3d_gt_bboxes_matched.json")
PRED_ROOT = os.path.join(ROOT, "output_test/nr3d")
SCAN_ROOT = os.path.join(ROOT, "data_eval/scannet")
OUT = os.path.join(ROOT, "ground_ral/nr3d/ablation/scorer_ablation_tacc.json")


def main():
    recs = [json.loads(line) for line in open(BALANCED) if line.strip()]
    gts = json.load(open(GT))
    gt_by_key = {}
    for g in gts:
        gt_by_key[(g.get("scene_id"), str(g.get("object_id")))] = g
        gt_by_key[(g.get("scene_id"), g.get("ann_id"))] = g
    pretrained = DEFAULT_CLIP_PRETRAINED
    if not os.path.isabs(str(pretrained)):
        pretrained = os.path.join(ROOT, str(pretrained))
    matcher = CLIPTextMatcher(DEFAULT_CLIP_MODEL, str(pretrained), "cpu")
    scene_cache = {}
    pred_cache = {}
    iou_cache = {}

    def get_scene(scene_id):
        if scene_id not in scene_cache:
            preds = json.load(open(os.path.join(PRED_ROOT, scene_id, "pred_bboxes.json")))
            scene_cache[scene_id] = SpatialSceneGraph.from_prediction_records(preds)
            pred_cache[scene_id] = {int(p["pred_id"]): p for p in preds if "pred_id" in p}
        return scene_cache[scene_id], pred_cache[scene_id]

    combos = ["full"] + [f"-{s}" for s in SCORER_NAMES] + ["-all_relations"]

    def disabled_for(combo):
        if combo == "full":
            return ()
        if combo == "-all_relations":
            return tuple(SCORER_NAMES)
        return (combo[1:],)

    picks = {combo: [] for combo in combos}
    dists = {combo: [] for combo in combos}
    correct = {combo: 0 for combo in combos}

    for rec in recs:
        scene_id = rec.get("scene_id")
        target = gt_by_key.get((scene_id, str(rec.get("object_id")))) or gt_by_key.get((scene_id, rec.get("ann_id")))
        if target is None:
            continue
        try:
            scene_graph, pred_map = get_scene(scene_id)
        except Exception:
            continue
        try:
            query_graph = QueryGraph.from_payload(rec["query_graph"])
        except Exception:
            continue
        grounder = QueryGraphGrounder(
            scene_graph, matcher,
            target_limit=24, ref_limit=10, beam_size=160, class_top_k=3,
        )
        for combo in combos:
            grounder.disabled_scorers = frozenset(disabled_for(combo))
            try:
                gres = grounder.ground(query_graph, top_k=1)
            except Exception:
                gres = {}
            res = gres.get("results") or []
            pid = int(res[0]["pred_id"]) if res else None
            dist = _compute_distance_for_prediction(target, pred_map, pid)
            sel = pred_map.get(pid) if pid is not None else None
            if sel is not None and _corresponds(sel, target, 0.5):
                correct[combo] += 1
            dists[combo].append(dist)
            picks[combo].append({
                "scene_id": scene_id,
                "object_id": int(rec.get("object_id")),
                "best_pred_id": pid,
            })

    rows = []
    n = len(picks["full"])
    for combo in combos:
        metrics = _summarize_dists(dists[combo])
        mean_dist = None if metrics is None else round(float(metrics[0]), 4)
        hit25 = hit50 = missing = no_pred = 0
        for rec in picks[combo]:
            scene_id = rec["scene_id"]
            if scene_id not in iou_cache:
                iou_cache[scene_id] = _scene_iou(scene_id, PRED_ROOT, SCAN_ROOT)
            scene = iou_cache[scene_id]
            if scene is None:
                missing += 1
                continue
            pid = rec["best_pred_id"]
            if pid is None:
                no_pred += 1
                continue
            if pid < 0 or pid >= scene["iou"].shape[0]:
                continue
            row = scene["iou"][pid]
            target = int(rec["object_id"])
            target_pos = np.where(scene["obj_ids"] == target)[0]
            target_iou = float(row[target_pos[0]]) if len(target_pos) else 0.0
            others = row.copy()
            if len(target_pos):
                others[target_pos[0]] = -1.0
            best_other = float(others.max()) if others.size else -1.0
            unique = target_iou > best_other
            if unique and target_iou >= 0.25:
                hit25 += 1
            if unique and target_iou >= 0.50:
                hit50 += 1
        scored = n - missing
        def pct(k):
            return round(100.0 * k / scored, 2) if scored else None
        sel = round(100.0 * correct[combo] / n, 2) if n else None
        rows.append({
            "combo": combo,
            "n": n,
            "scored": scored,
            "missing_scene": missing,
            "no_prediction": no_pred,
            "selacc": sel,
            "mean_dist": mean_dist,
            "tacc25": pct(hit25),
            "tacc50": pct(hit50),
            "hit25": hit25,
            "hit50": hit50,
        })
        print(
            f"{combo:16} sel={sel} mean={mean_dist} "
            f"T25={rows[-1]['tacc25']} T50={rows[-1]['tacc50']} "
            f"scored={scored} missing={missing} no_pred={no_pred}",
            flush=True,
        )
    full25 = rows[0]["tacc25"]
    full50 = rows[0]["tacc50"]
    for row in rows:
        row["delta25"] = None if row["tacc25"] is None else round(row["tacc25"] - full25, 2)
        row["delta50"] = None if row["tacc50"] is None else round(row["tacc50"] - full50, 2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"definition": "unique highest point-IoU; no selection incorrect; denominator excludes scenes whose masks did not load", "rows": rows}, f, indent=2)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
