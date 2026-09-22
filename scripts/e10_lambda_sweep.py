#!/usr/bin/env python3
"""
E10 — lambda_u re-sweep over NR3D/SR3D NS-Grounding records (replaces Table III).

R1: "Eq.(9) 不平衡" — the joint-matching score
      S = lambda_u * sum(s_node) + (1 - lambda_u) * sum(s_rel)
mixes node (CLIP semantic, usually ~[0.9,1.0]) and relation (geometric scorers,
often ~[0.0,0.8]) terms that live on different scales, so lambda_u barely acts
as a balance knob. E10 re-sweeps lambda_u under PER-TERM NORMALIZATION (each
term min-max normalized across candidates before the lambda_u-weighted sum) and
reports the same table as Table III (MeanDist / Acc@0.3m / Acc@0.5m), plus the
un-normalized sweep for reference.

This script replays stored query_graphs from a previous NS-Grounding output
jsonl against locally rebuilt scene graphs — ZERO extra LLM calls. Each
(lambda_u, normalize) cell rebuilds QueryGraphGrounder with
  graph_balance=lambda_u, normalize_terms=normalize
and scores the grounder's top-1 prediction with the standard NS protocol
(class-matched centroid <= corr_thresh).

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/e10_lambda_sweep.py \
      --outputs ground_ral/nr3d/qwen3_outputs_y.json \
      --gt nr3d/nr3d_gt_bboxes_matched.json \
      --pred-root output_test/nr3d \
      --lambdas 0.0,0.25,0.5,0.75,1.0 \
      --out ground_ral/nr3d/e10_lambda_sweep.json
"""
import argparse
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
    _compute_distance_for_prediction,
    _corresponds,
    _summarize_dists,
)
from spatial_scene_graph import SpatialSceneGraph  # noqa: E402

_DEVICE = os.environ.get("NS_CLIP_DEVICE") or "cpu"


def _load_records(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="NS run jsonl (records with query_graph)")
    ap.add_argument("--gt", required=True, help="nr3d/sr3d gt_bboxes_matched.json")
    ap.add_argument("--pred-root", required=True, help="dir with sceneXXX/pred_bboxes.json")
    ap.add_argument("--corr-thresh", type=float, default=0.5)
    ap.add_argument("--lambdas", default="0.0,0.25,0.5,0.75,1.0",
                    help="comma-separated lambda_u values (node-side share)")
    ap.add_argument("--clip-device", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "ground_ral", "nr3d", "e10_lambda_sweep.json"))
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    recs = _load_records(args.outputs)
    usable = [r for r in recs if isinstance(r.get("query_graph"), dict)]
    print(f"records: {len(recs)}, with query_graph: {len(usable)}, lambdas: {lambdas}")

    gts = json.load(open(args.gt))
    gt_by_key = {}
    for g in gts:
        gt_by_key[(g.get("scene_id"), str(g.get("object_id")))] = g
        gt_by_key[(g.get("scene_id"), g.get("ann_id"))] = g

    clip_matcher = CLIPTextMatcher(DEFAULT_CLIP_MODEL, str(DEFAULT_CLIP_PRETRAINED),
                                   args.clip_device or _DEVICE)
    scene_cache = {}
    pred_cache = {}

    def get_scene(scene_id: str):
        if scene_id not in scene_cache:
            pf = os.path.join(args.pred_root, scene_id, "pred_bboxes.json")
            preds = json.load(open(pf))
            scene_cache[scene_id] = SpatialSceneGraph.from_prediction_records(preds)
            pred_cache[scene_id] = {int(p["pred_id"]): p for p in preds if "pred_id" in p}
        return scene_cache[scene_id], pred_cache[scene_id]

    cells = [("none", False), ("norm", True)]
    results = {f"lambda={l}:{mode}": [] for l in lambdas for mode, _ in cells}
    pids = {f"lambda={l}:{mode}": [] for l in lambdas for mode, _ in cells}  # E10-C
    processed = 0

    for rec in usable:
        scene_id = rec.get("scene_id")
        target = gt_by_key.get((scene_id, str(rec.get("object_id")))) or \
                 gt_by_key.get((scene_id, rec.get("ann_id")))
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
        processed += 1
        gt_objs = {str(g.get("object_id")): g for g in gts if g.get("scene_id") == scene_id}
        covered = any(_corresponds(p, target, args.corr_thresh) for p in pred_map.values())

        for l in lambdas:
            for mode, norm in cells:
                grounder = QueryGraphGrounder(
                    scene_graph, clip_matcher,
                    target_limit=24, ref_limit=10, beam_size=160, class_top_k=3,
                    graph_balance=l, normalize_terms=norm,
                )
                try:
                    gres = grounder.ground(query_graph, top_k=1)
                except Exception:
                    gres = {}
                res = (gres.get("results") or [])
                pid = int(res[0]["pred_id"]) if res else None
                dist = _compute_distance_for_prediction(target, pred_map, pid)
                results[f"lambda={l}:{mode}"].append(dist)
                pids[f"lambda={l}:{mode}"].append(pid)

    print(f"\n== E10 lambda_u sweep over {processed} queries (grounder top-1, no rerank) ==")
    hdr = f"{'cell':<18}{'meanDist':>9}{'Acc@0.3':>9}{'Acc@0.5':>9}{'SelAcc':>9}"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for l in lambdas:
        for mode, _ in cells:
            key = f"lambda={l}:{mode}"
            metrics = _summarize_dists(results[key])
            if metrics is None or metrics[3] == 0:
                continue
            mean_dist, acc30, acc50, _, _ = metrics
            sel_all = sum(1 for d in results[key] if d is not None and d <= args.corr_thresh) / max(len(results[key]), 1)
            rows[key] = {"lambda_u": l, "normalize": mode,
                         "mean_dist": round(float(mean_dist), 4),
                         "acc03m": round(float(acc30), 4),
                         "acc05m": round(float(acc50), 4),
                         "selacc": round(float(sel_all), 4),
                         "queries": len(results[key])}
            print(f"{key:<18}{rows[key]['mean_dist']:>9.4f}{rows[key]['acc03m']:>9.4f}"
                  f"{rows[key]['acc05m']:>9.4f}{rows[key]['selacc']:>9.4f}")

    out = {"n_queries": processed, "rows": rows,
           "params": {"corr_thresh": args.corr_thresh, "lambdas": lambdas}}

    # ---- E10-C: decision-change rate (norm vs none) -------------------------
    print("\n== E10-C decision change rate: per-term normalization vs raw (same lambda_u) ==")
    print(f"{'lambda_u':>9}{'changed':>10}{'pct':>9}{'n':>8}")
    print("-" * 36)
    change_rows = {}
    for l in lambdas:
        a = pids[f"lambda={l}:none"]
        b = pids[f"lambda={l}:norm"]
        n = min(len(a), len(b))
        if n == 0:
            continue
        changed = sum(1 for i in range(n) if a[i] != b[i])
        change_rows[str(l)] = {"changed": changed, "n": n, "pct": round(changed / n, 4)}
        print(f"{l:>9}{changed:>10}{changed / n * 100:>8.1f}%{n:>8}")
    out["decision_change"] = change_rows

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
