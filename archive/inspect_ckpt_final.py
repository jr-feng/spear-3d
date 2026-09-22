#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def read_ply_points(path: Path) -> np.ndarray:
    """Minimal PLY reader to get xyz vertices; ignores list properties (faces)."""
    with path.open("rb") as f:
        header_lines = []
        fmt = None
        vert_count = None
        props = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF before end_header")
            line_dec = line.decode("utf-8").strip()
            header_lines.append(line_dec)
            if line_dec.startswith("format"):
                fmt = line_dec.split()[1]
            if line_dec.startswith("element vertex"):
                vert_count = int(line_dec.split()[-1])
                in_vertex = True
                continue
            if line_dec.startswith("element") and not line_dec.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line_dec.startswith("property"):
                parts = line_dec.split()
                if parts[1] == "list":
                    continue
                props.append((parts[2], parts[1]))
            if line_dec == "end_header":
                header_len = f.tell()
                break
    if fmt is None or vert_count is None:
        raise ValueError("Invalid PLY header")
    if fmt == "ascii":
        data = np.genfromtxt(path, skip_header=len(header_lines), max_rows=vert_count)
        return data[:, :3].astype(np.float32)
    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format {fmt}")
    np_types = {
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "uchar": "u1",
        "uint8": "u1",
        "int": "i4",
        "int32": "i4",
        "short": "i2",
        "uint": "u4",
        "uint32": "u4",
    }
    dtype = []
    for name, t in props:
        if t not in np_types:
            raise ValueError(f"Unsupported property type {t}")
        dtype.append((name, np_types[t]))
    with path.open("rb") as f:
        f.seek(header_len)
        arr = np.fromfile(f, count=vert_count, dtype=np.dtype(dtype))
    return np.vstack([arr["x"], arr["y"], arr["z"]]).T.astype(np.float32)


def summarize_npz(npz: np.lib.npyio.NpzFile) -> None:
    print(f"Keys: {list(npz.keys())}")
    for k in npz.keys():
        arr = npz[k]
        if arr.size == 0:
            arr_min = arr_max = "N/A"
        elif np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            arr_min = arr.min()
            arr_max = arr.max()
        else:
            arr_min = arr_max = "N/A (non-numeric)"
        print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr_min}, max={arr_max}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect ckpt_final.npz and compute per-mask centers.")
    ap.add_argument("--npz_path", type=Path, default="output_test/scannet_sam3_test/scene0011_00/ckpt_final.npz", help="Path to ckpt_final.npz")
    ap.add_argument(
        "--ply",
        type=Path,
        default="output_test/scannet_sam3_test/scene0011_00/final.ply",
        help="Path to final.ply (defaults to sibling final.ply)",
    )
    ap.add_argument("--max", type=int, default=None, help="Max instances to print (default: all)")
    args = ap.parse_args()

    npz_path = args.npz_path
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    ply_path = args.ply
    if ply_path is None:
        ply_path = npz_path.parent / "final.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    npz = np.load(npz_path, allow_pickle=True)
    print(f"Loaded {npz_path}")
    summarize_npz(npz)

    if "pred_masks" not in npz:
        print("No pred_masks found, skip center computation.")
        return

    pred_masks = npz["pred_masks"]
    if pred_masks.dtype == object:
        pred_masks = np.stack(pred_masks, axis=0)

    points = read_ply_points(ply_path)
    print(f"Loaded points from {ply_path}: {points.shape}")

    if pred_masks.shape[0] == points.shape[0]:
        masks = pred_masks
    elif pred_masks.shape[1] == points.shape[0]:
        masks = pred_masks.T
        print("Note: pred_masks transposed to match point count.")
    else:
        raise ValueError(
            f"pred_masks shape {pred_masks.shape} does not match points {points.shape[0]}"
        )

    num_inst = masks.shape[1]
    pred_classes = npz.get("pred_classes", None)
    pred_scores = npz.get("pred_score", None)
    pred_colors_true = npz.get("pred_colors_true", None)
    print(f"Instances: {num_inst}")

    max_print = num_inst if args.max is None else min(num_inst, args.max)
    for i in range(max_print):
        mask = masks[:, i].astype(bool)
        idx = np.where(mask)[0]
        count = int(idx.shape[0])
        if count == 0:
            center = None
        else:
            center = points[idx].mean(axis=0).tolist()
        cls_val = pred_classes[i] if pred_classes is not None and i < len(pred_classes) else None
        score_val = pred_scores[i] if pred_scores is not None and i < len(pred_scores) else None
        color_val = None
        if pred_colors_true is not None and i < len(pred_colors_true):
            color_val = pred_colors_true[i]
            if isinstance(color_val, np.ndarray):
                color_val = color_val.tolist()
        print(
            f"[{i:03d}] count={count}, center={center}, class={cls_val}, score={score_val}, color_rgb={color_val}"
        )

    if max_print < num_inst:
        print(f"... truncated, total {num_inst} instances")


if __name__ == "__main__":
    main()
