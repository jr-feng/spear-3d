#!/usr/bin/env python3
"""
E10-A — term-scale diagnosis: prove the Eq.(9) node/relation imbalance.

R1 ("Eq.(9) 不平衡"): the compatibility score adds a node term s_node (CLIP
semantic consistency, mapped to [0,1] and typically saturated near 0.9-1.0)
to a relation term s_rel (geometric scorer outputs, often far below 1). Adding
them with a single weight lambda_u only makes sense if both live on a
comparable scale, so the original lambda_u sweep may have been measuring
"saturation of CLIP" rather than a real node/relation balance trade-off.

This script collects, over every query in a stored NS-Grounding output jsonl
(zero extra LLM calls), per-candidate statistics for:
  * state level     : node_part  (geometric mean of node scores)
                      rel_part   (geometric mean of relation scores)
  * component level : every node unary component (class, attributes, ...)
                      every relation scorer score, grouped by rel_type
and writes a markdown/JSON statistics table plus histograms (PNG).

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/e10_term_distribution.py \
      --outputs ground_ral/nr3d/qwen3_outputs_y.json \
      --gt nr3d/nr3d_gt_bboxes_matched.json \
      --pred-root output_test/nr3d \
      --tag nr3d_qwen3 \
      --out-dir ground_ral/e10_diag
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from LLM_sam3_query_graph_clip import (  # noqa: E402
    CLIPTextMatcher,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    QueryGraph,
    QueryGraphGrounder,
)
from spatial_scene_graph import SpatialSceneGraph  # noqa: E402

_DEVICE = os.environ.get("NS_CLIP_DEVICE") or "cpu"


def _stats(vals):
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "median": round(float(np.median(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "max": round(float(a.max()), 4),
        "frac_ge_0.9": round(float((a >= 0.9).mean()), 4),
        "frac_le_0.5": round(float((a <= 0.5).mean()), 4),
    }


def _fmt_table(title, rows):
    lines = [f"### {title}", "",
             "| group | n | mean | std | min | p05 | median | p95 | max | frac≥0.9 | frac≤0.5 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, st in rows:
        if st is None:
            lines.append(f"| {name} | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {st['n']} | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} | "
            f"{st['p05']:.4f} | {st['median']:.4f} | {st['p95']:.4f} | {st['max']:.4f} | "
            f"{st['frac_ge_0.9']*100:.1f}% | {st['frac_le_0.5']*100:.1f}% |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="NS run jsonl (records with query_graph)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--tag", default="run", help="label for outputs, e.g. nr3d_qwen3")
    ap.add_argument("--graph-balance", type=float, default=0.5,
                    help="lambda_u used only to instantiate the grounder (distribution itself is lambda-free)")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--clip-device", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "ground_ral", "e10_diag"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.outputs) if l.strip()]
    usable = [r for r in recs if isinstance(r.get("query_graph"), dict)]
    if args.max_queries:
        usable = usable[: args.max_queries]
    gts = json.load(open(args.gt))
    gt_by_key = {}
    for g in gts:
        gt_by_key[(g.get("scene_id"), str(g.get("object_id")))] = g
        gt_by_key[(g.get("scene_id"), g.get("ann_id"))] = g

    clip_matcher = CLIPTextMatcher(DEFAULT_CLIP_MODEL, str(DEFAULT_CLIP_PRETRAINED),
                                   args.clip_device or _DEVICE)
    scene_cache = {}

    def get_scene(scene_id):
        if scene_id not in scene_cache:
            preds = json.load(open(os.path.join(args.pred_root, scene_id, "pred_bboxes.json")))
            scene_cache[scene_id] = SpatialSceneGraph.from_prediction_records(preds)
        return scene_cache[scene_id]

    node_part_vals, rel_part_vals = [], []
    comp_vals = defaultdict(list)        # unary component name -> values
    rel_vals = defaultdict(list)         # rel_type -> values
    n_states = n_queries = 0

    for rec in usable:
        scene_id = rec.get("scene_id")
        if (scene_id, str(rec.get("object_id"))) not in gt_by_key and \
           (scene_id, rec.get("ann_id")) not in gt_by_key:
            continue
        try:
            scene_graph = get_scene(scene_id)
        except Exception:
            continue
        try:
            qg = QueryGraph.from_payload(rec["query_graph"])
        except Exception:
            continue
        grounder = QueryGraphGrounder(
            scene_graph, clip_matcher,
            target_limit=24, ref_limit=10, beam_size=160, class_top_k=3,
            graph_balance=args.graph_balance,
        )
        try:
            gres = grounder.ground(qg, top_k=1000)  # all surviving target candidates
        except Exception:
            continue
        results = gres.get("results") or []
        if not results:
            continue
        n_queries += 1
        for r in results:
            n_states += 1
            if r.get("node_part") is not None:
                node_part_vals.append(float(r["node_part"]))
            if r.get("rel_part") is not None:
                rel_part_vals.append(float(r["rel_part"]))
            for _nid, ns in (r.get("node_scores") or {}).items():
                for cname, cval in (ns.get("components") or {}).items():
                    comp_vals[cname].append(float(cval))
            for _key, rs in (r.get("relation_scores") or {}).items():
                rel_vals[rs.get("type", "?")].append(float(rs.get("score", 0.0)))
        if n_queries % 25 == 0:
            print(f"  processed {n_queries} queries, {n_states} candidate states")

    # ---------------- report ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    node_st, rel_st = _stats(node_part_vals), _stats(rel_part_vals)
    md = [f"# E10-A term-scale diagnosis — {args.tag}",
          "",
          f"- queries: {n_queries}, candidate states: {n_states}",
          f"- grounder used for collection: lambda_u={args.graph_balance}, top_k=all",
          "",
          "State level: `node_part` = geometric mean of node (semantic) scores; "
          "`rel_part` = geometric mean of relation (geometric scorer) scores.", ""]
    md += _fmt_table("State-level distributions", [("node_part (s_node)", node_st),
                                                   ("rel_part (s_rel)", rel_st)])
    md.append("")
    md += _fmt_table("Node unary components (class / attributes)",
                     [(f"node:{k}", _stats(v)) for k, v in sorted(comp_vals.items())])
    md.append("")
    md += _fmt_table("Relation scorers by rel_type",
                     [(f"rel:{k}", _stats(v)) for k, v in sorted(rel_vals.items())])
    md.append("")

    if node_st and rel_st:
        gap = node_st["median"] - rel_st["median"]
        md += [f"**Imbalance summary**: median s_node = {node_st['median']:.3f} vs "
               f"median s_rel = {rel_st['median']:.3f} (gap = {gap:.3f}); "
               f"{node_st['frac_ge_0.9']*100:.1f}% of node scores are ≥0.9 while "
               f"{rel_st['frac_le_0.5']*100:.1f}% of relation scores are ≤0.5.", ""]

    out_json = os.path.join(args.out_dir, f"e10_diag_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "n_queries": n_queries, "n_states": n_states,
                   "graph_balance": args.graph_balance,
                   "state": {"node_part": node_st, "rel_part": rel_st},
                   "node_components": {k: _stats(v) for k, v in comp_vals.items()},
                   "relation_types": {k: _stats(v) for k, v in rel_vals.items()}},
                  f, ensure_ascii=False, indent=2)

    # ---------------- histograms ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(node_part_vals, bins=40, range=(0, 1), color="#2b6cb0")
    axes[0].set_title(f"s_node (node_part), n={len(node_part_vals)}")
    axes[0].set_xlabel("score"); axes[0].set_ylabel("#candidate states")
    axes[1].hist(rel_part_vals, bins=40, range=(0, 1), color="#c05621")
    axes[1].set_title(f"s_rel (rel_part), n={len(rel_part_vals)}")
    axes[1].set_xlabel("score")
    all_cls = comp_vals.get("class", [])
    axes[2].hist(all_cls, bins=40, range=(0, 1), color="#2f855a")
    axes[2].set_title(f"node 'class' component, n={len(all_cls)}")
    axes[2].set_xlabel("CLIP-derived class score")
    fig.suptitle(f"E10-A term-scale diagnosis — {args.tag} "
                 f"(node≈[{node_st['median']:.2f}] vs rel≈[{rel_st['median']:.2f}] median)")
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"e10_diag_{args.tag}.png")
    fig.savefig(png, dpi=140)

    md_path = os.path.join(args.out_dir, f"e10_diag_{args.tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"saved -> {md_path}\n         {out_json}\n         {png}")


if __name__ == "__main__":
    main()
