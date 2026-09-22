#!/usr/bin/env python3
"""
E9 / experiment A — synthetic-degradation robustness sweep (offline, ScanNet).

Orchestrates the whole robustness study against the reviewer point
"no evaluation under real sensor noise / motion blur / frame drop":

  1) gen     : build degraded frame copies   (scripts/degrade_frames.py)
  2) recon   : run the UNMODIFIED streaming pipeline on each degraded root
               (main_eval_scannet.py --arm A --dense-seg, 200-frame protocol)
  3) eval    : class-agnostic AP / class-aware AP / OVI-MAP mIoU-mAcc per condition
  4) report  : write a compact markdown/JSON summary table (baseline vs conditions)

Any single stage can be run alone (--stage gen|recon|eval|report), so long GPU
reconstruction runs can be launched by hand and evaluated later. Nothing here
touches sam3_service_1.py or the original data_eval/scannet frames.

Degradation conditions are declared as a table; weak/strong settings follow
ranges used in the robotics/online-mapping literature:

  blur     : motion-blur kernel length (px)     [5  = weak, 15 = strong]
  noise    : depth-noise sigma growth b in mm (sigma_mm = a + b*d_m^1.5),
             plus zeroed-pixel fraction          [b=0.002 weak, b=0.008 strong]
  drop     : uniform frame drop to N frames      [100 = -50%, 67 = -66%]

Baseline "none" is a plain 200-frame subset copy (verified byte-consistent
with e5_out_200_dense through the dataset loader), so every condition shares
the exact same 200-frame trajectory — differences are caused ONLY by the
injected degradation.

Usage (run each stage in the env that can see the SAM3 tunnel / GPUs):
  /root/anaconda3/envs/OASeg/bin/python scripts/e9_robustness.py \
      --stage gen --scenes scene0011_00,scene0432_00,scene0527_00 \
      --degrade-root data_eval_robust
  ... recon stage with the same args (needs --config/--device/--merge-gpu and
      the SAM3 service reachable on 127.0.0.1:18090, same as e5_out_200_dense)
  ... eval + report (CPU only)

Outputs:
  <degrade-root>/<cond>/<scene>/...            degraded frames
  <out-root>/<cond>/<scene>/{ckpt_final.npz,final.ply,...}   reconstructions
  <out-root>/e9_summary.md , e9_summary.json   aggregated table
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/root/anaconda3/envs/OASeg/bin/python"

# condition_id -> (degrade_kind, {kwargs}) ; see degrade_frames.py for kw names.
#
# Intensity ladder (mild / strong / severe). The previous 5px/15px blur and
# b=0.002/0.008 noise were too weak (noise floor masked the signal on 10
# scenes), so these are raised to produce a measurable, monotone drop.
#
#   blur   : --blur-len (px)  7 = mild, 21 = strong, 41 = severe
#   noise  : --noise-b        0.005 / 0.02 / 0.04  (sigma_mm = 0.5 + b*d_m^1.5)
#            --noise-drop     0.03 / 0.10 / 0.20   (fraction of depth holes)
#   drop   : --keep-frames    133 (6.7Hz) / 100 (5Hz) / 67 (3.3Hz) from 200
CONDITIONS = {
    "none":      ("none", {}),
    "blur_w":    ("blur", {"blur_len": 7}),
    "blur_s":    ("blur", {"blur_len": 21}),
    "blur_xs":   ("blur", {"blur_len": 41}),
    "noise_w":   ("noise", {"noise_b": 0.005, "noise_drop": 0.03}),
    "noise_s":   ("noise", {"noise_b": 0.02,  "noise_drop": 0.10}),
    "noise_xs":  ("noise", {"noise_b": 0.04,  "noise_drop": 0.20}),
    "drop133":   ("drop-uniform", {"keep_frames": 133}),
    "drop100":   ("drop-uniform", {"keep_frames": 100}),
    "drop67":    ("drop-uniform", {"keep_frames": 67}),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "recon", "eval", "report", "all"], required=True)
    ap.add_argument("--scenes", required=True, help="comma-separated scene names (e.g. scene0011_00,scene0432_00)")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated condition ids to run (default: all built-ins + any --cond-defs)")
    ap.add_argument("--cond-defs", nargs="*", default=None,
                    help="define / override conditions on the command line, no code edit needed. "
                         "Format per item: <id>:<kind>:k1=v1,k2=v2  (kind: none|blur|noise|"
                         "drop-uniform|drop-random). "
                         "Examples: "
                         "blur_11:blur:blur_len=11   "
                         "noise_m:noise:noise_b=0.005,noise_drop=0.03   "
                         "drop50:drop-random:keep_ratio=0.5,seed=0")
    ap.add_argument("--degrade-root", default=os.path.join(ROOT, "data_eval_robust"),
                    help="root holding degraded frame copies (gen output)")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "out_robust"),
                    help="root holding per-condition reconstructions + summary")
    ap.add_argument("--scans-root", default=os.path.join(ROOT, "data_eval", "scannet"),
                    help="original ScanNet root (never modified)")
    ap.add_argument("--baseline-dir", default=os.path.join(ROOT, "e5_out_200_dense"),
                    help="clean-baseline reconstruction root (same 200-frame dense protocol); "
                         "used in place of the 'none' condition for eval/report")
    # recon stage passthrough
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "scannet_test_1024.yaml"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge-gpu", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=200,
                    help="frame subset for none/blur/noise conditions (kept identical to e5 baseline)")
    return ap.parse_args()


def parse_cond_defs(defs):
    """'<id>:<kind>:k1=v1,k2=v2' -> {id: (kind, {k: typed_value})}"""
    out = {}
    kinds = {"none", "blur", "noise", "drop-uniform", "drop-random"}
    for item in defs or []:
        parts = item.strip().split(":")
        if len(parts) < 2:
            raise SystemExit(f"bad --cond-defs item '{item}' (need <id>:<kind>[:k=v,...])")
        cid, kind = parts[0].strip(), parts[1].strip()
        if kind not in kinds:
            raise SystemExit(f"bad kind '{kind}' in '{item}' (use one of {sorted(kinds)})")
        kw = {}
        if len(parts) > 2:
            for kv in parts[2].split(","):
                if not kv.strip():
                    continue
                k, _, v = kv.partition("=")
                k, v = k.strip(), v.strip()
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                kw[k] = v
        out[cid] = (kind, kw)
    return out


def cond_dir(args, cid):
    """Per-condition reconstruction root: none -> clean baseline dir (e5), else out-root/<cond>."""
    if cid == "none":
        return args.baseline_dir
    return os.path.join(args.out_root, cid)


def _run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage_gen(args, conds):
    print(f"\n=== E9 gen: build degraded copies for {len(conds)} conditions ===")
    scenes = " ".join(args.scenes.split(","))
    for cid, (kind, kw) in conds.items():
        cmd = [PY, os.path.join(ROOT, "scripts", "degrade_frames.py"),
               "--scans-root", args.scans_root,
               "--out-root", os.path.join(args.degrade_root, cid),
               "--degrade", kind,
               "--scenes"] + args.scenes.split(",")
        if kind in ("none", "blur", "noise"):
            cmd += ["--subsample", str(args.subsample)]
        for k, v in kw.items():
            flag = "--" + k.replace("_", "-")  # degrade_frames.py uses hyphenated flags
            cmd += [flag, str(v)]
        _run(cmd)


def stage_recon(args, conds):
    print(f"\n=== E9 recon: streaming reconstruction on degraded roots ===")
    scenes_csv = args.scenes  # comma separated ok for --scenes nargs='+'
    for cid in conds:
        out = os.path.join(args.out_root, cid)
        cmd = [PY, os.path.join(ROOT, "main_eval_scannet.py"),
               "--arm", "A",
               "--config", args.config,
               "--scans-root", os.path.join(args.degrade_root, cid),
               "--device", args.device,
               "--merge-gpu", str(args.merge_gpu),
               # MUST match the e5 baseline protocol exactly (dense per-frame
               # segmentation on the 200-frame subset) so degraded results are
               # comparable to e5_out_200_dense (= the clean 'none' row).
               "--dense-seg",
               "-o", out,
               "--scenes"] + scenes_csv.split(",")
        _run(cmd)


def stage_eval(args, conds):
    print(f"\n=== E9 eval: class-agnostic / class-aware / OVI-MAP per condition ===")
    gt_dir = args.scans_root
    scenes_csv = args.scenes
    result_dirs = [cond_dir(args, cid) for cid in conds]

    def ev(script, extra=None):
        cmd = [PY, os.path.join(ROOT, "scripts", script),
               "--result_dirs"] + result_dirs + \
              ["--gt_dir", gt_dir, "--scenes", scenes_csv]
        if extra:
            cmd += extra
        _run(cmd)

    ev("compare_ablation.py")
    ev("eval_class_aware.py", ["--tmp", os.path.join(args.out_root, "_cls")])
    ev("ovimap_aligned_eval.py", ["--tmp", os.path.join(args.out_root, "_ovimap")])


def stage_report(args, conds):
    print("\n=== E9 report: build summary table ===")
    scenes = args.scenes.split(",")
    gt_dir = args.scans_root
    rows = {}
    for cid in conds:
        rd = os.path.join(args.out_root, cid)
        rows[cid] = {"ap": None, "ap50": None, "ap25": None,
                     "cls_ap": None, "cls_ap50": None, "mIoU": None, "mAcc": None}
    # reuse the same evaluate functions in-process for compactness
    _real_argv = list(sys.argv)
    if os.path.join(ROOT, "eval") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "eval"))
    sys.argv = ["evaluate_seqs_scannet.py", "--result_dir", ".", "--gt_dir", "."]
    import evaluate_seqs_scannet as ev  # noqa: E402
    sys.argv = _real_argv

    import numpy as np
    agg = {}
    for cid in conds:
        rd = cond_dir(args, cid)
        preds, recons, gts, gps = [], [], [], []
        ok_scenes = []
        for seq in scenes:
            pf = os.path.join(rd, seq, "ckpt_final.npz")
            rf = os.path.join(rd, seq, "final.ply")
            gf = os.path.join(ROOT, "eval", "scannet200", "validation", seq + ".txt")
            gp = os.path.join(gt_dir, seq, seq + "_vh_clean_2.ply")
            if not all(os.path.isfile(x) for x in (pf, rf, gf, gp)):
                print(f"[skip] {cid}/{seq}: missing output/gt files")
                continue
            preds.append(pf); recons.append(rf); gts.append(gf); gps.append(gp)
            ok_scenes.append(seq)
        if not ok_scenes:
            continue
        results, _ = ev.evaluate(ok_scenes, preds, gts, recons, gps, output_file=None)
        if results:
            agg[cid] = {"scenes": len(results),
                        "ap": float(np.mean([r["ap"] for r in results])),
                        "ap50": float(np.mean([r["ap50"] for r in results])),
                        "ap25": float(np.mean([r["ap25"] for r in results]))}
    rows.update(agg)

    out_json = os.path.join(args.out_root, "e9_summary.json")
    os.makedirs(args.out_root, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"conditions": list(conds), "scenes": scenes, "rows": rows},
                  f, ensure_ascii=False, indent=2)

    lines = ["# E9 robustness summary (synthetic degradation, ScanNet200 val subset)",
             "",
             f"- scenes: {', '.join(scenes)}",
             f"- protocol: {args.subsample}-frame per-frame segmentation (--dense-seg), same as e5 baseline",
             f"- baseline 'none' = {args.baseline_dir} (clean e5 outputs, not re-run)",
             "",
             "| condition | AP | AP50 | AP25 |",
             "|---|---|---|---|"]
    order = (["none"] if "none" in conds else []) + [c for c in conds if c != "none"]
    for cid in order:
        r = rows.get(cid) or {}
        ap = r.get("ap"); ap50 = r.get("ap50"); ap25 = r.get("ap25")
        fmt = lambda x: "-" if x is None else f"{x:.4f}"
        lines.append(f"| {cid} | {fmt(ap)} | {fmt(ap50)} | {fmt(ap25)} |")
    md = os.path.join(args.out_root, "e9_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved -> {md} , {out_json}")


def main():
    args = parse_args()
    table = dict(CONDITIONS)
    table.update(parse_cond_defs(args.cond_defs))

    if args.conditions:
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    else:
        # default: built-ins (none first) + any custom conditions in def order
        conds = list(CONDITIONS.keys()) + [c for c in parse_cond_defs(args.cond_defs) if c not in CONDITIONS]

    unknown = [c for c in conds if c not in table]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}; available: {list(table)}")

    if args.stage == "all":
        # one command: gen -> recon -> eval -> report (recon is the only GPU/expensive pass;
        # each stage loops over ALL conditions once, so total recon runs = n_scenes x n_conds)
        stage_gen(args, {c: table[c] for c in conds})
        stage_recon(args, conds)
        stage_eval(args, conds)
        stage_report(args, conds)
    elif args.stage == "gen":
        stage_gen(args, {c: table[c] for c in conds})
    elif args.stage == "recon":
        stage_recon(args, conds)
    elif args.stage == "eval":
        stage_eval(args, conds)
    elif args.stage == "report":
        stage_report(args, conds)


if __name__ == "__main__":
    main()
