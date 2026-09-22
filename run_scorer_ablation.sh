#!/bin/bash
# ============================================================
# Automatic 7-scorer ablation for NS-Grounding (R2: "7 hand-crafted
# scorers have no ablation / no sufficiency argument").
#
# Pipeline:
#   1. collect existing qwen3-parsed pools (old500 + new100 if present)
#   2. supplement: parse more NR3D queries whose text hints at rare
#      relations (side/between/vert/...), so every scorer class has
#      >= K queries (parser = qwen3, --disable-rerank => cheap)
#   3. select a BALANCED subset (select_balanced.py): ~K queries per scorer
#   4. offline ablation replay (ablate_scorers.py, ZERO extra LLM calls)
#   5. print summary table
#
# Usage:
#   bash run_scorer_ablation.sh            # K=15 default
#   K=20 bash run_scorer_ablation.sh
# ============================================================
set -e
cd "$(dirname "$0")"
PY_OAS=/root/anaconda3/envs/OASeg/bin/python

GT_FILE=${GT_FILE:-/home/OnlineAnySeg/nr3d/nr3d_gt_bboxes_matched.json}
PRED_ROOT=${PRED_ROOT:-/home/OnlineAnySeg/output_test/nr3d}
K=${K:-15}
SUPP_PER_CLASS=${SUPP_PER_CLASS:-30}
WORK=ground_ral/nr3d/ablation
mkdir -p "$WORK"

POOL_OLD=/home/OnlineAnySeg/output_test/nr3d/qwen_outputs_y.json
POOL_NEW=/home/OnlineAnySeg/ground_ral/nr3d/qwen3_outputs_y.json
SUPP_OUT="$WORK/ns_supp_outputs.jsonl"

# ---- Step 1: per-scorer availability in existing pools ----
$PY_OAS - << EOF
import json, os
pools = ["$POOL_OLD"]
if os.path.isfile("$POOL_NEW"):
    pools.append("$POOL_NEW")
SCORER_REL = {"dir": ("left_of","right_of","front_of","behind"),
              "near": ("near",), "rank": ("closest_to","farthest_from"),
              "vert": ("above","below"), "contact": ("on_top_of","under"),
              "side": ("same_side_as",), "between": ("between",)}
REV = {rt: s for s, rels in SCORER_REL.items() for rt in rels}
from collections import Counter
cnt = Counter()
have = set()
for p in pools:
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        qg = r.get("query_graph")
        if not isinstance(qg, dict):
            continue
        have.add((r.get("scene_id"), str(r.get("ann_id"))))
        for rel in (qg.get("relations") or []):
            t = rel.get("type")
            if t in REV:
                cnt[REV[t]] += 1
need = {s: max(0, $K - cnt.get(s, 0)) for s in SCORER_REL}
todo = [s for s, n in need.items() if n > 0]
json.dump({"need": need, "todo": todo}, open("$WORK/need.json", "w"))
print("现有池每 scorer:", dict(cnt))
print("需补:", {s: n for s, n in need.items() if n > 0})
print("TODO 类:", todo)
EOF

TODO=$(/root/anaconda3/envs/OASeg/bin/python -c "import json;print(' '.join(json.load(open('$WORK/need.json'))['todo']))")

# ---- Step 2: keyword candidates + parse supplementation ----
if [ -n "$TODO" ]; then
    echo "== 补 parse 稀有关系类: [$TODO] (每类 ≤$SUPP_PER_CLASS 条关键词候选)"
    echo "  [内部] 生成关键词候选并 parse..."
    /root/anaconda3/envs/OASeg/bin/python - "$TODO" << 'EOF'
import json, re, os, sys
GT_FILE = "/home/OnlineAnySeg/nr3d/nr3d_gt_bboxes_matched.json"
WORK = "ground_ral/nr3d/ablation"
todo = sys.argv[1].split()
kw = {
  "dir": re.compile(r'\b(left|right|front|behind|in front)\b'),
  "near": re.compile(r'\b(near|next to|beside|adjacent|close to)\b'),
  "rank": re.compile(r'\b(closest|nearest|farthest|furthest)\b'),
  "vert": re.compile(r'\b(above|below|under|over)\b'),
  "contact": re.compile(r'\b(on top of|standing on|sitting on|on the)\b'),
  "side": re.compile(r'\b(same side|along the wall|against the wall|by the wall|near the wall)\b'),
  "between": re.compile(r'\bbetween\b'),
}
gt = json.load(open(GT_FILE))
have = set()
for p in ("/home/OnlineAnySeg/output_test/nr3d/qwen_outputs_y.json",
          "/home/OnlineAnySeg/ground_ral/nr3d/qwen3_outputs_y.json"):
    if os.path.isfile(p):
        for line in open(p):
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                have.add((r.get("scene_id"), str(r.get("ann_id"))))
            except Exception: pass
SUPP = int(os.environ.get("SUPP_PER_CLASS", "30"))
merged = {}
for s in todo:
    rx = kw[s]
    sel = [g for g in gt if rx.search((g.get("description") or "").lower())
           and (g.get("scene_id"), str(g.get("ann_id"))) not in have]
    for g in sel[:SUPP]:
        merged[(g.get("scene_id"), str(g.get("ann_id")))] = g
    print(f"  {s}: 关键词候选 {min(len(sel), SUPP)} 条")
json.dump(list(merged.values()), open(f"{WORK}/supp_merge.json", "w"), ensure_ascii=False)
print(f"  补池合并: {len(merged)} 条 -> {WORK}/supp_merge.json")
EOF

    echo "== NS parse-only (qwen3, --disable-rerank, clip cpu) =="
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY $PY_OAS LLM_sam3_query_graph_clip.py \
        --config config/grounding_eval/grounding_eval_qwen3_y_nr3d.yaml \
        --gt-path "$WORK/supp_merge.json" \
        --disable-rerank --clip-device cpu \
        --out "$SUPP_OUT" \
        --metrics-out "$WORK/ns_supp_metrics.json" 2>&1 | tee "$WORK/out_supp.log" || echo "[WARN] 补 parse 有失败, 继续"
else
    echo "== 现有池已满足每类 $K 条, 无需补 parse"
fi

# ---- Step 3: balanced selection ----
POOL_ARGS="$POOL_OLD"
[ -f "$POOL_NEW" ] && POOL_ARGS="$POOL_ARGS $POOL_NEW"
[ -f "$SUPP_OUT" ] && POOL_ARGS="$POOL_ARGS $SUPP_OUT"
echo "== 均衡抽样 (每 scorer $K 条) =="
$PY_OAS select_balanced.py --pools $POOL_ARGS \
    --per-scorer "$K" --out "$WORK/ns_ablation_balanced.jsonl"

# ---- Step 4: offline ablation replay ----
echo "== 离线消融重放 (9 组合, 零额外 LLM) =="
$PY_OAS ablate_scorers.py \
    --outputs "$WORK/ns_ablation_balanced.jsonl" \
    --gt "$GT_FILE" \
    --pred-root "$PRED_ROOT" \
    --corr-thresh 0.5 \
    --out "$WORK/scorer_ablation.json" 2>&1 | tee "$WORK/out_ablate.log"

echo "== DONE. 结果: $WORK/scorer_ablation.json"
