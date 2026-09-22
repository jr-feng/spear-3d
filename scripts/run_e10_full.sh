#!/usr/bin/env bash
# =============================================================================
# E10 full pipeline — answers R1 ⑨ "Eq.(9) 不平衡 + 求解算法未说明"
#
#   A. term-scale diagnosis  (scripts/e10_term_distribution.py)
#      -> proves s_node (CLIP, saturated ~0.9-1.0) vs s_rel (geometric, ~0-0.8)
#   B. full lambda_u double-sweep, none vs per-term-norm
#      (scripts/e10_lambda_sweep.py) -> replaces Table III, incl. Acc@0.3m
#   C. decision-change rate norm-vs-none at the same lambda_u (from B)
#   D. backbone robustness: lambda_u=0.5 across qwen3 / deepseek / GLM (from B)
#
# PREREQUISITE (full scale): NS-Grounding outputs with per-record query_graph.
#   The repo ships 100-query smoke outputs. For the paper Table III query set
#   (CSVG-same-set: NR3D 1450 / SR3D 5247) generate them first:
#
#     sed -i 's/^MAX_GT = 500/MAX_GT = 20000/' LLM_sam3_query_graph_clip.py
#     # set eval.max in the yaml to your query-set size, then:
#     /root/anaconda3/envs/OASeg/bin/python LLM_sam3_query_graph_clip.py \
#         --config config/grounding_eval/grounding_eval_qwen3_y_nr3d.yaml --disable-rerank
#     /root/anaconda3/envs/OASeg/bin/python LLM_sam3_query_graph_clip.py \
#         --config config/grounding_eval/grounding_eval_qwen3_y_sr3d.yaml --disable-rerank
#   (--disable-rerank is enough: E10 replays the parsed query_graph only.)
#
# Usage:
#   bash scripts/run_e10_full.sh                                  # smoke (100 queries)
#   bash scripts/run_e10_full.sh --ns-nr3d <full_nr3d.jsonl> --ns-sr3d <full_sr3d.jsonl>
# =============================================================================
set -u
PY=/root/anaconda3/envs/OASeg/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDAS="0.0,0.25,0.5,0.75,1.0"
CORR=0.5
NS_NR3D="$ROOT/ground_ral/nr3d/qwen3_outputs_y.json"
NS_SR3D="$ROOT/ground_ral/sr3d/qwen3_y_outputs.jsonl"
# backbone robustness variants (label:path); smoke files shipped in repo
VARIANTS="qwen3:$ROOT/ground_ral/nr3d/qwen3_outputs_y.json,deepseek:$ROOT/ground_ral/nr3d/deepseek_y_outputs.jsonl,glm:$ROOT/ground_ral/nr3d/glm_y_outputs.jsonl"

while [ $# -gt 0 ]; do
  case "$1" in
    --ns-nr3d) NS_NR3D="$2"; shift 2 ;;
    --ns-sr3d) NS_SR3D="$2"; shift 2 ;;
    --lambdas) LAMBDAS="$2"; shift 2 ;;
    --corr-thresh) CORR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

have() { [ -n "${1:-}" ] && [ -f "$1" ] && [ "$(wc -l < "$1")" -gt 0 ]; }
nrec() { wc -l < "$1" | tr -d ' '; }

echo "=============================================================="
echo " E10 full pipeline   lambdas=$LAMBDAS  corr<=$CORR"
echo "=============================================================="

# ---------------- A: term-scale diagnosis ----------------
echo
echo "--- [A] term-scale diagnosis ---"
DIAG_SETS="nr3d_qwen3|$NS_NR3D|nr3d/nr3d_gt_bboxes_matched.json"
IFS=',' read -ra VARR <<< "$VARIANTS"
for v in "${VARR[@]}"; do
  label="${v%%:*}"; path="${v#*:}"
  have "$path" && DIAG_SETS="$DIAG_SETS;nr3d_${label}|$path|nr3d/nr3d_gt_bboxes_matched.json"
done
if have "$NS_SR3D"; then
  DIAG_SETS="$DIAG_SETS;sr3d_qwen3|$NS_SR3D|sr3d/sr3d_gt_bboxes_matched.json"
fi

IFS=';' read -ra DS <<< "$DIAG_SETS"
for d in "${DS[@]}"; do
  IFS='|' read -r tag path gt <<< "$d"
  echo "[A] $tag  <- $path"
  "$PY" "$ROOT/scripts/e10_term_distribution.py" \
      --outputs "$path" --gt "$ROOT/$gt" --pred-root "$ROOT/output_test/nr3d" \
      --tag "$tag" --out-dir "$ROOT/ground_ral/e10_diag" || echo "[A] FAILED: $tag"
done

# ---------------- B: sweeps ----------------
run_sweep() {  # path gt tag outjson
  local path="$1" gt="$2" tag="$3" out="$4"
  if ! have "$path"; then
    echo "[B] SKIP $tag: missing NS output '$path'"
    echo "    (generate it first — see header of this script)"
    return 0
  fi
  local n; n=$(nrec "$path")
  echo "[B] $tag: $n records"
  [ "$n" -lt 500 ] && echo "[B] WARNING: $n < 500 records — SMOKE scale, not the full Table III set"
  mkdir -p "$(dirname "$out")"
  "$PY" "$ROOT/scripts/e10_lambda_sweep.py" \
      --outputs "$path" --gt "$ROOT/$gt" --pred-root "$ROOT/output_test/nr3d" \
      --lambdas "$LAMBDAS" --corr-thresh "$CORR" --out "$out" || echo "[B] FAILED: $tag"
}

echo
echo "--- [B] lambda_u sweep (none vs per-term norm) ---"
run_sweep "$NS_NR3D" nr3d/nr3d_gt_bboxes_matched.json "nr3d" "$ROOT/ground_ral/nr3d/e10_lambda_sweep_full.json"
run_sweep "$NS_SR3D" sr3d/sr3d_gt_bboxes_matched.json "sr3d" "$ROOT/ground_ral/sr3d/e10_lambda_sweep_full.json"

# ---------------- D: backbone robustness at lambda_u=0.5 ----------------
echo
echo "--- [D] backbone robustness (lambda_u=0.5, none vs norm) ---"
for v in "${VARR[@]}"; do
  label="${v%%:*}"; path="${v#*:}"
  run_sweep "$path" nr3d/nr3d_gt_bboxes_matched.json "nr3d_${label}" \
            "$ROOT/ground_ral/nr3d/e10_lambda_sweep_${label}.json"
done

# ---------------- report ----------------
echo
echo "--- report ---"
cat > "$ROOT/ground_ral/_e10_report.py" <<'PYEOF'
import json, os
ROOT = "/home/OnlineAnySeg"

def J(p):
    return json.load(open(p)) if os.path.isfile(p) else None

md = ["# E10 — Eq.(9) term balance: evidence, fix, and re-sweep", "",
      "Grounder top-1, rerank disabled (replay of stored query_graphs; no extra LLM calls). "
      "`none` = original lambda_u-weighted fusion; `norm` = node/relation terms min-max "
      "normalized across candidates before lambda_u weighting.", ""]

# ---- A table ----
diag_dir = f"{ROOT}/ground_ral/e10_diag"
diags = []
if os.path.isdir(diag_dir):
    for f in sorted(os.listdir(diag_dir)):
        if f.endswith(".json"):
            d = J(os.path.join(diag_dir, f))
            if d and d.get("state", {}).get("node_part"):
                diags.append(d)
if diags:
    md += ["## A. Term-scale diagnosis (imbalance proof)", "",
           "| run | queries | states | median s_node | median s_rel | gap | s_node≥0.9 | s_rel≤0.5 |",
           "|---|---|---|---|---|---|---|---|"]
    for d in diags:
        n, r = d["state"]["node_part"], d["state"]["rel_part"]
        md.append(f"| {d['tag']} | {d['n_queries']} | {d['n_states']} | {n['median']:.3f} | "
                  f"{r['median']:.3f} | {n['median']-r['median']:.3f} | "
                  f"{n['frac_ge_0.9']*100:.1f}% | {r['frac_le_0.5']*100:.1f}% |")
    md.append("")

# ---- B/C tables ----
for tag, jp in [("NR3D", f"{ROOT}/ground_ral/nr3d/e10_lambda_sweep_full.json"),
                ("SR3D", f"{ROOT}/ground_ral/sr3d/e10_lambda_sweep_full.json")]:
    j = J(jp)
    if not j or not j.get("rows"):
        md += [f"## B. {tag} lambda_u sweep", "", "_not run (missing full-scale NS output)_", ""]
        continue
    rows = j["rows"]
    md += [f"## B. {tag} lambda_u sweep ({j['n_queries']} queries)", "",
           "| λu | MeanDist (none) | Acc@0.3 (none) | Acc@0.5 (none) | MeanDist (norm) | Acc@0.3 (norm) | Acc@0.5 (norm) |",
           "|---|---|---|---|---|---|---|"]
    for l in sorted({r["lambda_u"] for r in rows.values()}):
        n, m = rows.get(f"lambda={l}:none"), rows.get(f"lambda={l}:norm")
        f_ = lambda r, k: "-" if not r else f"{r[k]:.4f}"
        md.append(f"| {l} | {f_(n,'mean_dist')} | {f_(n,'acc03m')} | {f_(n,'acc05m')} | "
                  f"{f_(m,'mean_dist')} | {f_(m,'acc03m')} | {f_(m,'acc05m')} |")
    md.append("")
    dc = j.get("decision_change")
    if dc:
        md += ["### C. Decision-change rate (norm vs none, same λu)", "",
               "| λu | changed top-1 | total | rate |", "|---|---|---|---|"]
        for l in sorted(dc, key=float):
            md.append(f"| {l} | {dc[l]['changed']} | {dc[l]['n']} | {dc[l]['pct']*100:.1f}% |")
        md.append("")

# ---- D table ----
md += ["## D. Backbone robustness at λu = 0.5 (NR3D)", "",
       "| backbone | Acc@0.5 (none) | Acc@0.5 (norm) | Acc@0.3 (none) | Acc@0.3 (norm) |",
       "|---|---|---|---|---|"]
for label in ["qwen3", "deepseek", "glm"]:
    j = J(f"{ROOT}/ground_ral/nr3d/e10_lambda_sweep_{label}.json")
    if not j or not j.get("rows"):
        md.append(f"| {label} | - | - | - | - |")
        continue
    n, m = j["rows"].get("lambda=0.5:none"), j["rows"].get("lambda=0.5:norm")
    f_ = lambda r, k: "-" if not r else f"{r[k]:.4f}"
    md.append(f"| {label} | {f_(n,'acc05m')} | {f_(m,'acc05m')} | {f_(n,'acc03m')} | {f_(m,'acc03m')} |")
md.append("")

out = f"{ROOT}/ground_ral/E10_report.md"
open(out, "w").write("\n".join(md) + "\n")
print("saved ->", out)
PYEOF
"$PY" "$ROOT/ground_ral/_e10_report.py"

echo
echo "E10 pipeline done."
echo "  diagnosis : ground_ral/e10_diag/*.md|json|png"
echo "  sweeps    : ground_ral/{nr3d,sr3d}/e10_lambda_sweep_full.json"
echo "  robustness: ground_ral/nr3d/e10_lambda_sweep_{qwen3,deepseek,glm}.json"
echo "  report    : ground_ral/E10_report.md"
