#!/usr/bin/env python3
"""
E3 vocabulary-scale benchmark client (local, run with OASeg env).

Reads a few frames from data_eval/scannet/<scene>/color/*.jpg, calls the server
(/e3, e.g. via ssh tunnel 127.0.0.1:18091 -> container:8091) with vocabulary
subsets of size V in {20, 50, 100, 200}, and aggregates:
  * per-frame inference latency  -> FPS_effective
  * peak GPU memory per call     (server-reported incremental peak)
  * segment count / classes      (coarse semantic sanity)

Vocabularies: top-V ScanNet200 classes by frequency over the GT validation txts
(eval/scannet200/validation), or sequential top-V if --vocab-mode seq.

Usage:
  python e3_vocab_bench.py \
      --server http://127.0.0.1:18091 \
      --scenes scene0011_00 scene0015_00 \
      --frames 5 --vocabs 20,50,100,200 --repeat 3 \
      --out ground_ral/e3_vocab_bench.json
"""
import argparse
import base64
import io
import json
import os
import statistics
import sys
import time

import numpy as np
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load_vocab_freq(top: int, gt_dir: str = "eval/scannet200/validation") -> list:
    """top-V ScanNet200 labels by frequency in GT validation instance txts."""
    from eval.scannet200.scannet200_constants import CLASS_LABELS_200  # noqa: E402
    import collections
    cnt = collections.Counter()
    for fn in os.listdir(gt_dir):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(gt_dir, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cnt[int(line) // 1000] += 1
                except ValueError:
                    pass
    ordered = [cid for cid, _ in cnt.most_common() if 0 <= cid < len(CLASS_LABELS_200)]
    labels = [CLASS_LABELS_200[cid] for cid in ordered]
    # fill to top if fewer than top
    for cid in range(len(CLASS_LABELS_200)):
        if len(labels) >= top:
            break
        if cid not in cnt:
            labels.append(CLASS_LABELS_200[cid])
    return labels[:top]


def load_vocab_seq(top: int) -> list:
    from eval.scannet200.scannet200_constants import CLASS_LABELS_200  # noqa: E402
    return list(CLASS_LABELS_200)[:top]


def frame_paths(scene: str, scans_root: str, n: int) -> list:
    color_dir = os.path.join(scans_root, scene, "color")
    names = sorted(os.listdir(color_dir), key=lambda x: int(x.split(".")[0]))
    return [os.path.join(color_dir, names[i]) for i in range(min(n, len(names)))]


def b64_of(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18091")
    ap.add_argument("--scans-root", default="/home/OnlineAnySeg/data_eval/scannet")
    ap.add_argument("--scenes", nargs="+", default=["scene0011_00", "scene0015_00"])
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--vocabs", default="20,50,100,200")
    ap.add_argument("--repeat", type=int, default=3, help="repeats after 1 warmup")
    ap.add_argument("--vocab-mode", choices=["freq", "seq"], default="freq")
    ap.add_argument("--out", default="ground_ral/e3_vocab_bench.json")
    args = ap.parse_args()

    vocabs = [int(v) for v in args.vocabs.split(",") if v.strip()]
    url = args.server.rstrip("/") + "/e3"

    # warm frames per scene
    frames = []
    for sc in args.scenes:
        paths = frame_paths(sc, args.scans_root, args.frames)
        frames.append((sc, [b64_of(p) for p in paths]))
        print(f"scene {sc}: {len(paths)} frames")

    vocab_map = {}
    for v in vocabs:
        vocab_map[v] = (load_vocab_freq(v) if args.vocab_mode == "freq" else load_vocab_seq(v))

    rows = []
    for v in vocabs:
        vocab = vocab_map[v]
        lat_all, mem_all, seg_all = [], [], []
        oom_hits = 0
        for sc, b64s in frames:
            for bi, b64 in enumerate(b64s):
                # warmup once per (v, first frame)
                for rep in range(args.repeat + (1 if (bi == 0) else 0)):
                    payload = {"image_base64": b64, "text_prompts": vocab, "return_map": False}
                    t0 = time.time()
                    r = requests.post(url, json=payload, timeout=300)
                    dt = time.time() - t0
                    try:
                        j = r.json()
                    except Exception:
                        print(f"[{sc}] bad resp for V={v}: {r.status_code} {r.text[:200]}")
                        continue
                    if rep == 0 and bi == 0:
                        continue  # warmup discard
                    if j.get("oom"):
                        oom_hits += 1
                        continue
                    if j.get("error"):
                        print(f"[{sc}] err V={v}: {j['error']}")
                        continue
                    lat_all.append(j["processing_time_sec"])
                    mem_all.append(j["peak_mem_mb"])
                    seg_all.append(j["segments_count"])
        if not lat_all:
            print(f"V={v}: no successful calls (oom={oom_hits})")
            rows.append({"vocab": v, "latency_ms": None, "fps": None,
                         "peak_mem_mb": None, "segments_avg": None, "oom": oom_hits})
            continue
        med_ms = statistics.median(lat_all) * 1000.0
        med_mem = statistics.median(mem_all)
        avg_seg = statistics.mean(seg_all)
        rows.append({
            "vocab": v,
            "latency_ms": round(med_ms, 2),
            "fps": round(1000.0 / med_ms, 2) if med_ms > 0 else None,
            "peak_mem_mb": round(med_mem, 1),
            "segments_avg": round(avg_seg, 1),
            "n_calls": len(lat_all),
            "oom": oom_hits,
        })
        print(f"V={v:>4}: median {med_ms:7.1f} ms | FPS_eff {1000.0/med_ms:6.2f} "
              f"| peak_mem {med_mem:7.1f} MB | segs ~{avg_seg:.0f} | oom {oom_hits}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "scenes": args.scenes, "frames": args.frames,
                   "vocab_mode": args.vocab_mode, "server": args.server}, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
