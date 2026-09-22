#!/usr/bin/env python3
"""
Offline scorer ablation + relation-coverage analysis for NS-Grounding.

Zero extra LLM calls: replays the stored `query_graph` of every record from a
previous NS-Grounding run against locally rebuilt scene graphs, with each of
the 7 hand-crafted relation scorers (paper Sec. III-B.4) disabled one at a
time, plus a "no relation scorers at all" variant (class/CLIP node matching
only). The grounder's top-1 prediction is scored with the same NS protocol
(class-matched centroid <= 0.5 m) used everywhere else, so the resulting rows
are directly comparable with the CSVG / L0 tables.

It also prints a relation-coverage analysis over the same queries: how often
each rel_type / scorer fires, and the share of queries whose relations are
fully expressible by the 7-scorer library (sufficiency argument for R2).

Usage:
  python ablate_scorers.py \
      --outputs output_test/nr3d/qwen_outputs_y.json \
      --gt nr3d/nr3d_gt_bboxes_matched.json \
      --pred-root output_test/nr3d \
      --corr-thresh 0.5 \
      --out ground_ral/nr3d/scorer_ablation.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from LLM_sam3_query_graph_clip import (  # noqa: E402
    CLIPTextMatcher,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    QueryGraph,
    QueryGraphGrounder,
    SCORER_NAMES,
    SCORER_RELTYPES,
    _compute_distance_for_prediction,
    _corresponds,
    _summarize_dists,
)
from spatial_scene_graph import SpatialSceneGraph  # noqa: E402

import os

_DEVICE = os.environ.get("NS_CLIP_DEVICE") or "cpu"  # text-only matching is fine on cpu


def _load_records(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="NS run jsonl (records with query_graph)")
    ap.add_argument("--gt", required=True, help="nr3d/sr3d gt_bboxes_matched.json")
    ap.add_argument("--pred-root", required=True, help="dir with sceneXXX/pred_bboxes.json")
    ap.add_argument("--corr-thresh", type=float, default=0.5)
    ap.add_argument("--clip-device", default=None)
    ap.add_argument("--out", default="ground_ral/nr3d/scorer_ablation.json")
    args = ap.parse_args()

    recs = _load_records(args.outputs)
    usable = [r for r in recs if isinstance(r.get("query_graph"), dict)]
    print(f"records: {len(recs)}, with query_graph: {len(usable)}")

    # ---- GT index (same as eval_csvg_ns) ----
    gts = json.load(open(args.gt))
    gt_by_key = {}
    for g in gts:
        gt_by_key[(g.get("scene_id"), str(g.get("object_id")))] = g
        gt_by_key[(g.get("scene_id"), g.get("ann_id"))] = g

    # ---- caches ----
    clip_matcher = CLIPTextMatcher(DEFAULT_CLIP_MODEL, str(DEFAULT_CLIP_PRETRAINED),
                                   _DEVICE)
    scene_cache: dict = {}
    pred_cache: dict = {}

    def get_scene(scene_id: str):
        if scene_id not in scene_cache:
            pf = os.path.join(args.pred_root, scene_id, "pred_bboxes.json")
            preds = json.load(open(pf))
            scene_cache[scene_id] = SpatialSceneGraph.from_prediction_records(preds)
            pred_cache[scene_id] = {int(p["pred_id"]): p for p in preds if "pred_id" in p}
        return scene_cache[scene_id], pred_cache[scene_id]

    combos = ["full"] + [f"-{s}" for s in SCORER_NAMES] + ["-all_relations"]

    def disabled_for(combo: str):
        if combo == "full":
            return ()
        if combo == "-all_relations":
            return tuple(SCORER_NAMES)
        return (combo[1:],)

    # relation coverage counters (independent of combo)
    rel_freq: Counter = Counter()
    scorer_query_hits: Counter = Counter()   # queries whose relations touch scorer s
    n_no_rel = n_parse_fail = 0

    results: dict = {combo: [] for combo in combos}  # combo -> list of dist
    correct_cnt: dict = {combo: 0 for combo in combos}
    covered_cnt: dict = {combo: 0 for combo in combos}
    no_pred_cnt: dict = {combo: 0 for combo in combos}
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
            n_parse_fail += 1
            continue

        # relation coverage per query
        rel_types = [rel.rel_type for rel in query_graph.relations]
        if not rel_types:
            n_no_rel += 1
        for rt in rel_types:
            rel_freq[rt] += 1
        for rt in rel_types:
            for name, rels in SCORER_RELTYPES.items():
                if rt in rels:
                    scorer_query_hits[name] += 1

        processed += 1
        gt_objs = {}
        for g in gts:
            if g.get("scene_id") == scene_id:
                gt_objs[str(g.get("object_id"))] = g
        covered = any(_corresponds(p, target, args.corr_thresh) for p in pred_map.values())

        for combo in combos:
            grounder = QueryGraphGrounder(
                scene_graph, clip_matcher,
                target_limit=24, ref_limit=10, beam_size=160, class_top_k=3,
                disabled_scorers=disabled_for(combo),
            )
            try:
                gres = grounder.ground(query_graph, top_k=1)
            except Exception:
                gres = {}
            res = (gres.get("results") or [])
            pid = int(res[0]["pred_id"]) if res else None
            dist = _compute_distance_for_prediction(target, pred_map, pid)
            sel = pred_map.get(pid) if pid is not None else None
            correct = False
            if sel is not None and _corresponds(sel, target, args.corr_thresh):
                correct = True
            if pid is None:
                no_pred_cnt[combo] += 1
            if covered:
                covered_cnt[combo] += 1
            if correct:
                correct_cnt[combo] += 1
            results[combo].append(dist)

    # ---- aggregate ----
    rows = []
    for combo in combos:
        metrics = _summarize_dists(results[combo])
        if metrics is None or metrics[3] == 0:
            continue
        mean_dist, acc30, acc50, valid_total, total = metrics
        sel_all = correct_cnt[combo] / max(processed, 1)
        rows.append({
            "combo": combo,
            "disabled": list(disabled_for(combo)) or "none",
            "processed": processed,
            "valid": valid_total,
            "no_prediction": no_pred_cnt[combo],
            "mean_dist": round(float(mean_dist), 4),
            "acc03m_pred": round(float(acc30), 4),
            "acc05m_pred": round(float(acc50), 4),
            "acc05m_all": round(sel_all, 4),
            "selacc_all": round(sel_all, 4),
            "coverage": round(covered_cnt[combo] / max(processed, 1), 4),
        })

    # ---- print ----
    print(f"\n== Scorer ablation over {processed} queries (grounder top-1, no rerank) ==")
    hdr = f"{'combo':<14}{'acc05@pred':>10}{'acc05@all':>10}{'no_pred':>8}{'meanDist':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['combo']:<14}{r['acc05m_pred']:>10.4f}{r['acc05m_all']:>10.4f}{r['no_prediction']:>8}{r['mean_dist']:>9.3f}")

    print("\n== Relation coverage (sufficiency for R2) ==")
    print(f"queries with query_graph: {processed}, parse failures: {n_parse_fail}, no relations: {n_no_rel}")
    print("rel_type frequency:")
    for rt, n in rel_freq.most_common():
        print(f"  {rt:<14}{n:>6}  ({100.0 * n / max(processed, 1):.1f}% of queries)")
    print("scorer coverage (queries touching >=1 rel of the scorer):")
    for s in SCORER_NAMES:
        n = scorer_query_hits[s]
        print(f"  {s:<10}{n:>6}  ({100.0 * n / max(processed, 1):.1f}%)")

    out = {
        "n_queries": processed,
        "n_parse_fail": n_parse_fail,
        "n_no_relation": n_no_rel,
        "rows": rows,
        "rel_freq": dict(rel_freq),
        "scorer_query_hits": dict(scorer_query_hits),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
