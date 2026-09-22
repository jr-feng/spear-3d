#!/usr/bin/env python3
"""
Balanced relation-subset selector for scorer ablation.

From one or more NS-run jsonl pools (records with `query_graph`), selects a
subset in which every one of the 7 relation scorers is exercised by roughly
`--per-scorer` queries. Selection greedily prefers queries that touch a single
scorer (clean attribution); multi-scorer queries are used only as fillers.

Usage:
  python select_balanced.py \
      --pools output_test/nr3d/qwen_outputs_y.json ground_ral/nr3d/qwen3_outputs_y.json \
      --pool-names old500 new100 \
      --per-scorer 15 \
      --out ground_ral/nr3d/ns_ablation_balanced.jsonl
"""
import argparse
import json
import sys
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from LLM_sam3_query_graph_clip import SCORER_RELTYPES, SCORER_NAMES  # noqa: E402

_REV = {rt: s for s, rels in SCORER_RELTYPES.items() for rt in rels}


def load_pool(path: str):
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    out = []
    for r in recs:
        qg = r.get("query_graph")
        if not isinstance(qg, dict):
            continue
        rels = [x.get("type") for x in (qg.get("relations") or [])]
        scorers = frozenset(_REV.get(x) for x in rels if x in _REV)
        out.append((r, scorers, rels))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--per-scorer", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = []
    for p in args.pools:
        recs.extend(load_pool(p))
    # de-dup on (scene_id, ann_id)
    seen = {}
    for r, scorers, rels in recs:
        key = (r.get("scene_id"), str(r.get("ann_id")))
        if key not in seen:
            seen[key] = (r, scorers, rels)
    all_recs = list(seen.values())
    print(f"候选池去重后: {len(all_recs)} 条")

    # counters
    cnt = Counter()
    for _, scorers, _ in all_recs:
        for s in scorers:
            cnt[s] += 1
    print("候选池 scorer 命中:", dict(cnt))

    single = defaultdict(list)   # scorer -> recs touching ONLY that scorer
    multi = []
    for item in all_recs:
        _, scorers, _ = item
        if len(scorers) == 1:
            single[next(iter(scorers))].append(item)
        elif len(scorers) > 1:
            multi.append(item)

    # ---- greedy selection: rare scorers first, fill with multi-scorer queries ----
    need = {s: args.per_scorer for s in SCORER_NAMES}
    chosen_keys = set()
    chosen: list = []

    # pass 1: single-scorer queries
    for s in sorted(SCORER_NAMES, key=lambda x: cnt.get(x, 0)):
        for item in single.get(s, []):
            if need[s] <= 0:
                break
            r, scorers, _ = item
            key = (r.get("scene_id"), str(r.get("ann_id")))
            if key in chosen_keys:
                continue
            chosen_keys.add(key)
            chosen.append(item)
            need[s] -= 1

    # pass 2: multi-scorer fillers
    for item in multi:
        r, scorers, _ = item
        # count how many scorers still need this query
        hungry = [s for s in scorers if need[s] > 0]
        if not hungry:
            continue
        key = (r.get("scene_id"), str(r.get("ann_id")))
        if key in chosen_keys:
            continue
        chosen_keys.add(key)
        chosen.append(item)
        for s in hungry:
            need[s] -= 1

    # ---- stats ----
    final_cnt = Counter()
    rel_cnt = Counter()
    for _, scorers, rels in chosen:
        for s in scorers:
            final_cnt[s] += 1
        for rt in rels:
            rel_cnt[rt] += 1

    print("\n== 均衡子集 ==")
    print(f"总查询: {len(chosen)}")
    print("每 scorer 命中:", {s: final_cnt.get(s, 0) for s in SCORER_NAMES})
    print("rel_type 分布:", dict(rel_cnt.most_common()))
    unmet = {s: v for s, v in need.items() if v > 0}
    if unmet:
        print(f"[WARN] 以下 scorer 不足 {args.per_scorer} 条: {unmet}")
        print("       需要补充 parse 含这些关系的查询后重试")
    else:
        print("每类均达到目标 ✓")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for item, _, _ in chosen:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n子集 -> {args.out}")


if __name__ == "__main__":
    main()
