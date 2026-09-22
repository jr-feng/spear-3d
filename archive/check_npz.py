#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ contents")
    parser.add_argument("npz_path", type=Path, help="Path to .npz file")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Optional: directory to dump each array as .npy")
    args = parser.parse_args()

    data = np.load(args.npz_path)
    print(f"Loaded {args.npz_path}")
    print(f"Keys: {list(data.keys())}")

    for k in data.keys():
        arr = data[k]
        # Safely compute min/max only for numeric/bool arrays
        if arr.size == 0:
            arr_min = arr_max = "N/A"
        elif np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            arr_min = arr.min()
            arr_max = arr.max()
        else:
            arr_min = arr_max = "N/A (non-numeric)"

        print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr_min}, max={arr_max}")
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.save_dir / f"{args.npz_path.stem}_{k}.npy"
            np.save(out_path, arr)
            print(f"  saved to {out_path}")

    # If prediction arrays exist, list classes and scores side by side
    if "pred_classes" in data and "pred_score" in data:
        classes = data["pred_classes"]
        scores = data["pred_score"]
        print("\nPredictions (index: class, score):")
        for i, (c, s) in enumerate(zip(classes, scores)):
            print(f"{i:03d}: class={c}, score={s}")

if __name__ == "__main__":
    main()
