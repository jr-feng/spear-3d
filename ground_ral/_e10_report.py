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
