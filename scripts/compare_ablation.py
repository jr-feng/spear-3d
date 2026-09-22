#!/usr/bin/env python3
"""
消融对比评估：对多个重建结果目录跑同一套 eval，输出并排 AP/AP50/AP25 对比表。

用法（用 OASeg 环境）：
  /root/anaconda3/envs/OASeg/bin/python scripts/compare_ablation.py \
      --result_dirs output/ablation_sam3 output/ablation_cropformer \
      --gt_dir /workspace/nr3d/scans \
      --scenes scene0011_00,scene0015_00,scene0025_00,scene0030_00
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dirs", nargs="+", required=True,
                   help="重建结果目录（可多个），每个目录含 sceneXXX/ckpt_final.npz 与 final.ply")
    p.add_argument("--gt_dir", required=True, help="GT pc 目录，如 /workspace/nr3d/scans")
    p.add_argument("--gt_seg_dir", default=os.path.join(ROOT, "eval", "scannet200", "validation"),
                   help="GT 语义分割 .txt 目录")
    p.add_argument("--scenes", default=None, help="逗号分隔的场景名；默认=第一个目录下的所有场景")
    return p.parse_args()


def collect(result_dir, scenes, ev, args):
    if scenes:
        seqs = [s.strip() for s in scenes.split(",") if s.strip()]
    else:
        seqs = sorted(d for d in os.listdir(result_dir)
                      if os.path.isdir(os.path.join(result_dir, d)))

    pred_files, gt_files, recon_pcs, gt_pcs = [], [], [], []
    for seq in seqs:
        pf = os.path.join(result_dir, seq, "ckpt_final.npz")
        rf = os.path.join(result_dir, seq, "final.ply")
        gf = os.path.join(args.gt_seg_dir, seq + ".txt")
        gp = os.path.join(args.gt_dir, seq, seq + "_vh_clean_2.ply")
        if not all(os.path.isfile(x) for x in (pf, rf, gf, gp)):
            print(f"[skip] {result_dir}/{seq}: 缺少文件")
            continue
        pred_files.append(pf); recon_pcs.append(rf)
        gt_files.append(gf); gt_pcs.append(gp)

    # evaluate() 内部自行 read_point_cloud，返回 scene_results, failed_scenes
    results, _ = ev.evaluate(seqs[:len(pred_files)], pred_files, gt_files, recon_pcs, gt_pcs, output_file=None)
    return {r["name"]: r for r in results}, seqs


def main():
    args = parse_args()
    sys.argv = ["evaluate_seqs_scannet.py", "--result_dir", args.result_dirs[0], "--gt_dir", args.gt_dir]
    import evaluate_seqs_scannet as ev

    all_scenes = []
    per_dir = {}
    for rd in args.result_dirs:
        rmap, seqs = collect(rd, args.scenes, ev, args)
        per_dir[rd] = rmap
        all_scenes.extend(seqs)

    # 去重保持顺序
    seen, scenes = set(), []
    for s in all_scenes:
        if s not in seen:
            seen.add(s); scenes.append(s)

    hdr = f"{'scene':<16}" + "".join(f"{os.path.basename(rd):>28}" for rd in args.result_dirs)
    print("\n" + hdr)
    sub = " " * 16 + "".join(f"{'AP':>9}{'AP50':>10}{'AP25':>9}" for _ in args.result_dirs)
    print(sub)

    sums = {rd: np.zeros(3) for rd in args.result_dirs}
    cnts = {rd: 0 for rd in args.result_dirs}
    for s in scenes:
        row = f"{s:<16}"
        for rd in args.result_dirs:
            r = per_dir[rd].get(s)
            if r is None:
                row += " " * 28
            else:
                row += f"{r['ap']:>9.4f}{r['ap50']:>10.4f}{r['ap25']:>9.4f}"
                sums[rd] += np.array([r['ap'], r['ap50'], r['ap25']])
                cnts[rd] += 1
        print(row)

    row = f"{'average':<16}"
    for rd in args.result_dirs:
        if cnts[rd]:
            a = sums[rd] / cnts[rd]
            row += f"{a[0]:>9.4f}{a[1]:>10.4f}{a[2]:>9.4f}"
        else:
            row += " " * 28
    print(row)


if __name__ == "__main__":
    main()
