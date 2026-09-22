#!/usr/bin/env python3
"""
Generate one large paper-style figure per successful NR3D example.

Design goal:
- one figure = one result
- point-cloud/mask views are the visual focus
- minimal text
- no text boxes covering the point-cloud panels
"""

from __future__ import annotations

import sys, os  # allow running from paper/ subdir: add repo root to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from spatial_scene_graph import SpatialSceneGraph


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "output_test" / "nr3d" / "deepseek_outputs_y.json"
DEFAULT_GT = ROOT / "nr3d" / "nr3d_gt_bboxes_matched.json"
DEFAULT_SCENE_ROOT = ROOT / "output_test" / "nr3d"
DEFAULT_OUTDIR = ROOT / "figs" / "query_graph_nr3d_single_figs"

SUCCESS_DISTANCE_THRESH = 0.5
BACKGROUND_POINT_LIMIT = 70000
BACKGROUND_POINT_LIMIT_3D = 150000
MASK_POINT_LIMIT = 40000
MASK_POINT_LIMIT_3D = 80000

COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#1F1F1F"
COLOR_TARGET = "#C44536"
COLOR_REF = "#2D6A9F"
COLOR_GT = "#2E8B57"
COLOR_MUTED = "#7F7F7F"
COLOR_TOPK = ["#C44536", "#F39C4A", "#E9C46A", "#84A98C", "#577590"]

FONT_BASE = 18
FONT_SMALL = 14      # ≈ 0.78x
FONT_BODY = 16       # ≈ 0.9x
FONT_PANEL = 22      # ≈ 1.2x
FONT_HEADER = 26     # ≈ 1.45x
FONT_HEADER_BODY = 16

def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_BASE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

            "svg.fonttype": "none",

            "axes.titlesize": FONT_PANEL,
            "axes.labelsize": FONT_BASE,

            "axes.facecolor": COLOR_BG,
            "figure.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,

            "axes.edgecolor": "#444444",
            "axes.labelcolor": COLOR_TEXT,

            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "xtick.labelsize": FONT_BASE,
            "ytick.labelsize": FONT_BASE,

            "legend.fontsize": FONT_BASE,
            "text.color": COLOR_TEXT,
        }
    )


def _wrap(text: str, width: int) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_slug(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:120] or "sample"


def _shorten(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)] + "…"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse line {lineno} in {path}: {exc}") from exc
    return records


def _build_gt_index(gt_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in data:
        scene_id = str(item.get("scene_id") or "")
        if not scene_id:
            continue
        ref_id = item.get("ref_id")
        ann_id = item.get("ann_id")
        if ref_id is not None:
            index[(scene_id, f"ref:{ref_id}")] = item
        if ann_id is not None:
            index[(scene_id, f"ann:{ann_id}")] = item
    return index


def _lookup_gt(record: Mapping[str, Any], gt_index: Mapping[Tuple[str, str], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scene_id = str(record.get("scene_id") or "")
    ref_id = record.get("ref_id")
    ann_id = record.get("ann_id")
    if scene_id and ref_id is not None:
        hit = gt_index.get((scene_id, f"ref:{ref_id}"))
        if hit is not None:
            return hit
    if scene_id and ann_id is not None:
        hit = gt_index.get((scene_id, f"ann:{ann_id}"))
        if hit is not None:
            return hit
    return None


def _read_ply_points_colors(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with path.open("rb") as f:
        header_lines = []
        fmt = None
        vert_count = None
        props: List[Tuple[str, str]] = []
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
    if fmt != "binary_little_endian" or vert_count is None:
        raise ValueError(f"Unsupported or invalid PLY: {path}")
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
    for name, typ in props:
        if typ not in np_types:
            raise ValueError(f"Unsupported property type {typ} in {path}")
        dtype.append((name, np_types[typ]))
    with path.open("rb") as f:
        f.seek(header_len)
        arr = np.fromfile(f, count=vert_count, dtype=np.dtype(dtype))
    points = np.vstack([arr["x"], arr["y"], arr["z"]]).T.astype(np.float32)
    colors = None
    if {"red", "green", "blue"}.issubset(arr.dtype.names or []):
        colors = np.vstack([arr["red"], arr["green"], arr["blue"]]).T.astype(np.float32)
        if colors.max() > 1.5:
            colors = colors / 255.0
    return points, colors


def _bbox_from_record(record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    box_min = record.get("box_min")
    box_max = record.get("box_max")
    if isinstance(box_min, (list, tuple)) and isinstance(box_max, (list, tuple)) and len(box_min) == 3 and len(box_max) == 3:
        return np.asarray(box_min, dtype=np.float64), np.asarray(box_max, dtype=np.float64)
    center = np.asarray(record.get("center"), dtype=np.float64)
    size = np.asarray(record.get("size"), dtype=np.float64)
    half = size / 2.0
    return center - half, center + half


def _center_from_record(record: Mapping[str, Any]) -> np.ndarray:
    center = record.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 3:
        return np.asarray(center, dtype=np.float64)
    box_min, box_max = _bbox_from_record(record)
    return (box_min + box_max) / 2.0


def _project_box(box_min: np.ndarray, box_max: np.ndarray, plane: str) -> Tuple[float, float, float, float]:
    if plane == "xy":
        return float(box_min[0]), float(box_min[1]), float(box_max[0] - box_min[0]), float(box_max[1] - box_min[1])
    if plane == "xz":
        return float(box_min[0]), float(box_min[2]), float(box_max[0] - box_min[0]), float(box_max[2] - box_min[2])
    raise ValueError(f"Unsupported plane: {plane}")


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _highlight_pred_ids(record: Mapping[str, Any], max_rank: int) -> Tuple[List[int], List[int], Optional[int]]:
    top_results = record.get("graph_top_results") or []
    assignments = top_results[0].get("assignments") if top_results else {}
    ref_pred_ids: List[int] = []
    if isinstance(assignments, dict):
        for node_id, payload in assignments.items():
            if node_id == "t":
                continue
            pred_id = payload.get("pred_id")
            if pred_id is not None:
                ref_pred_ids.append(int(pred_id))

    topk_pred_ids: List[int] = []
    for item in top_results[:max_rank]:
        pred_id = item.get("pred_id")
        if pred_id is not None:
            topk_pred_ids.append(int(pred_id))

    best_pred_id = record.get("best_pred_id")
    return ref_pred_ids, topk_pred_ids, int(best_pred_id) if best_pred_id is not None else None


def _bbox_corners(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


def _draw_bbox_3d(ax: plt.Axes, box_min: np.ndarray, box_max: np.ndarray, *, color: str, linewidth: float, alpha: float) -> None:
    corners = _bbox_corners(box_min, box_max)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
        )


def _load_scene_data(scene_root: Path, scene_id: str) -> Dict[str, Any]:
    scene_dir = scene_root / scene_id
    preds = json.loads((scene_dir / "pred_bboxes.json").read_text(encoding="utf-8"))
    pred_map = {int(item["pred_id"]): dict(item) for item in preds}

    points, colors = _read_ply_points_colors(scene_dir / "final.ply")
    ckpt = np.load(scene_dir / "ckpt_final.npz", allow_pickle=True)
    masks = ckpt["pred_masks"]
    if masks.dtype == object:
        masks = np.stack(masks, axis=0)
    if masks.shape[0] != points.shape[0] and masks.shape[1] == points.shape[0]:
        masks = masks.T
    if masks.shape[0] != points.shape[0]:
        raise ValueError(f"Mask shape {masks.shape} does not match point count {points.shape[0]} in {scene_dir}")
    masks = masks.astype(bool)

    return {
        "scene_dir": scene_dir,
        "preds": preds,
        "pred_map": pred_map,
        "points": points,
        "colors": colors,
        "masks": masks,
        "room_min": points.min(axis=0),
        "room_max": points.max(axis=0),
        "scene_graph": SpatialSceneGraph.from_prediction_records(preds),
    }


def _assets_exist(scene_root: Path, scene_id: str) -> bool:
    scene_dir = scene_root / scene_id
    return all(
        path.is_file()
        for path in [
            scene_dir / "pred_bboxes.json",
            scene_dir / "final.ply",
            scene_dir / "ckpt_final.npz",
        ]
    )


def _success_key(record: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    qg = record.get("query_graph") or {}
    rels = len(qg.get("relations") or [])
    nodes = len(qg.get("nodes") or [])
    top_results = record.get("graph_top_results") or []
    top1 = _safe_float(top_results[0].get("total_score"), 0.0) if top_results else 0.0
    dist = _safe_float(record.get("distance"), 999.0)
    return (rels, nodes, top1, -dist)


def _select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scene_root: Path,
    success_distance: float,
    top_n: int,
    ref_ids: Sequence[int],
    scene_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    ref_filter = set(ref_ids)
    scene_filter = set(scene_ids)
    selected: List[Dict[str, Any]] = []
    for record in records:
        scene_id = str(record.get("scene_id") or "")
        if not scene_id:
            continue
        if ref_filter and record.get("ref_id") not in ref_filter:
            continue
        if scene_filter and scene_id not in scene_filter:
            continue
        if not _assets_exist(scene_root, scene_id):
            continue
        dist = record.get("distance")
        if not isinstance(dist, (int, float)) or float(dist) > success_distance:
            continue
        if record.get("graph_error") or record.get("parse_error"):
            continue
        if not record.get("graph_top_results") or record.get("best_pred_id") is None:
            continue
        selected.append(dict(record))
    selected.sort(key=_success_key, reverse=True)
    return selected[:top_n]


def _draw_query_graph(ax: plt.Axes, query_graph: Mapping[str, Any]) -> None:
    ax.set_title("Query graph", fontsize=FONT_PANEL, loc="left")
    ax.axis("off")
    nodes = query_graph.get("nodes") or []
    relations = query_graph.get("relations") or []
    if not nodes:
        ax.text(0.5, 0.5, "No query graph", ha="center", va="center")
        return

    target_nodes = [node for node in nodes if str(node.get("role")) == "target"]
    ref_nodes = [node for node in nodes if str(node.get("role")) != "target"]
    positions: Dict[str, Tuple[float, float]] = {}

    if target_nodes:
        positions[str(target_nodes[0].get("id") or "t")] = (0.25, 0.52)
    if ref_nodes:
        ys = np.linspace(0.20, 0.82, num=len(ref_nodes))
        for node, y in zip(ref_nodes, ys):
            positions[str(node.get("id") or "r")] = (0.78, float(y))

    for node in nodes:
        node_id = str(node.get("id") or "")
        x, y = positions.get(node_id, (0.5, 0.5))
        is_target = str(node.get("role")) == "target"
        edge = COLOR_TARGET if is_target else COLOR_REF
        face = "#F5D9D5" if is_target else "#DCE8F2"
        label = str(node.get("class") or "object")
        attrs = node.get("attributes") or []
        if attrs:
            label += f"\n{', '.join(map(str, attrs[:2]))}"
        patch = FancyBboxPatch(
            (x - 0.16, y - 0.11),
            0.32,
            0.22,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.7,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=FONT_BODY)

    for relation in relations:
        source = str(relation.get("source") or "")
        targets = relation.get("targets")
        if not isinstance(targets, list):
            single = relation.get("target")
            targets = [single] if single is not None else []
        sx, sy = positions.get(source, (0.5, 0.5))
        for target in targets:
            tx, ty = positions.get(str(target), (0.5, 0.5))
            arrow = FancyArrowPatch(
                (sx + 0.12, sy),
                (tx - 0.12, ty),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#6C6C6C",
            )
            ax.add_patch(arrow)
            ax.text(
                (sx + tx) / 2.0,
                (sy + ty) / 2.0 + 0.04,
                str(relation.get("type") or "rel"),
                ha="center",
                va="center",
                fontsize=FONT_SMALL,
                bbox=dict(boxstyle="round,pad=0.16", facecolor=COLOR_BG, edgecolor="#9A9A9A", linewidth=0.6),
            )


def _draw_pointcloud_panel(
    ax: plt.Axes,
    *,
    scene_data: Mapping[str, Any],
    record: Mapping[str, Any],
    gt_record: Optional[Mapping[str, Any]],
    plane: str,
    max_rank: int,
    rasterize_points: bool = True,
) -> None:
    points = scene_data["points"]
    colors = scene_data["colors"]
    masks = scene_data["masks"]
    room_min = scene_data["room_min"]
    room_max = scene_data["room_max"]

    if plane == "xy":
        coords = points[:, [0, 1]]
        title = "Point cloud + instance masks (XY)"
        ylabel = "y"
    else:
        coords = points[:, [0, 2]]
        title = "Point cloud + instance masks (XZ)"
        ylabel = "z"

    bg_idx = _sample_indices(points.shape[0], BACKGROUND_POINT_LIMIT)
    bg_coords = coords[bg_idx]
    bg_colors = colors[bg_idx] if colors is not None else np.full((len(bg_idx), 3), 0.62, dtype=np.float32)
    bg_colors = np.clip(bg_colors * 0.9, 0.0, 1.0)
    ax.scatter(
        bg_coords[:, 0],
        bg_coords[:, 1],
        s=2.2,
        c=bg_colors,
        alpha=0.55,
        linewidths=0.0,
        rasterized=rasterize_points,
        zorder=1,
    )

    ref_pred_ids, topk_pred_ids, best_pred_id = _highlight_pred_ids(record, max_rank)

    for pred_id in ref_pred_ids:
        if pred_id >= masks.shape[1]:
            continue
        idx = np.where(masks[:, pred_id])[0]
        if idx.size == 0:
            continue
        idx = idx[_sample_indices(idx.size, MASK_POINT_LIMIT)]
        c = coords[idx]
        ax.scatter(
            c[:, 0],
            c[:, 1],
            s=3.0,
            c=COLOR_REF,
            alpha=0.85,
            linewidths=0.0,
            rasterized=rasterize_points,
            zorder=2,
        )

    for rank, pred_id in enumerate(topk_pred_ids, 1):
        if pred_id >= masks.shape[1]:
            continue
        idx = np.where(masks[:, pred_id])[0]
        if idx.size == 0:
            continue
        idx = idx[_sample_indices(idx.size, MASK_POINT_LIMIT)]
        c = coords[idx]
        color = COLOR_TOPK[min(rank - 1, len(COLOR_TOPK) - 1)]
        alpha = 1.0 if pred_id == best_pred_id else 0.65
        size = 4.0 if pred_id == best_pred_id else 2.2
        ax.scatter(
            c[:, 0],
            c[:, 1],
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0.0,
            rasterized=rasterize_points,
            zorder=3,
        )

    if gt_record is not None:
        gt_box_min, gt_box_max = _bbox_from_record(gt_record)
        x, y, w, h = _project_box(gt_box_min, gt_box_max, plane)
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.8,
                edgecolor=COLOR_GT,
                facecolor="none",
                linestyle=(0, (4, 2)),
                zorder=4,
            )
        )

    x0 = float(room_min[0])
    x1 = float(room_max[0])
    if plane == "xy":
        y0 = float(room_min[1])
        y1 = float(room_max[1])
    else:
        y0 = float(room_min[2])
        y1 = float(room_max[2])
    pad_x = max((x1 - x0) * 0.04, 0.08)
    pad_y = max((y1 - y0) * 0.04, 0.08)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_title(title, fontsize=FONT_PANEL, loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.10, linestyle=":", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")


def _draw_pointcloud_panel_3d(
    ax: plt.Axes,
    *,
    scene_data: Mapping[str, Any],
    record: Mapping[str, Any],
    gt_record: Optional[Mapping[str, Any]],
    max_rank: int,
    rasterize_points: bool = True,
) -> None:
    points = scene_data["points"]
    colors = scene_data["colors"]
    masks = scene_data["masks"]
    pred_map = scene_data["pred_map"]
    room_min = scene_data["room_min"]
    room_max = scene_data["room_max"]

    bg_idx = _sample_indices(points.shape[0], BACKGROUND_POINT_LIMIT_3D)
    bg_points = points[bg_idx]
    bg_colors = colors[bg_idx] if colors is not None else np.full((len(bg_idx), 3), 0.62, dtype=np.float32)
    bg_colors = np.clip(bg_colors * 0.78, 0.0, 1.0)
    ax.scatter(
        bg_points[:, 0],
        bg_points[:, 1],
        bg_points[:, 2],
        s=1.10,
        c=bg_colors,
        alpha=0.34,
        linewidths=0.0,
        depthshade=False,
        rasterized=rasterize_points,
        zorder=1,
    )

    ref_pred_ids, topk_pred_ids, best_pred_id = _highlight_pred_ids(record, max_rank)

    for pred_id in ref_pred_ids:
        if pred_id >= masks.shape[1]:
            continue
        idx = np.where(masks[:, pred_id])[0]
        if idx.size == 0:
            continue
        idx = idx[_sample_indices(idx.size, MASK_POINT_LIMIT_3D)]
        c = points[idx]
        ax.scatter(
            c[:, 0],
            c[:, 1],
            c[:, 2],
            s=1.95,
            c=COLOR_REF,
            alpha=0.76,
            linewidths=0.0,
            depthshade=False,
            rasterized=rasterize_points,
            zorder=2,
        )

    for rank, pred_id in enumerate(topk_pred_ids, 1):
        if pred_id >= masks.shape[1]:
            continue
        idx = np.where(masks[:, pred_id])[0]
        if idx.size == 0:
            continue
        idx = idx[_sample_indices(idx.size, MASK_POINT_LIMIT_3D)]
        c = points[idx]
        color = COLOR_TOPK[min(rank - 1, len(COLOR_TOPK) - 1)]
        alpha = 0.98 if pred_id == best_pred_id else 0.64
        size = 3.10 if pred_id == best_pred_id else 1.95
        ax.scatter(
            c[:, 0],
            c[:, 1],
            c[:, 2],
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0.0,
            depthshade=False,
            rasterized=rasterize_points,
            zorder=3,
        )

    if best_pred_id is not None and int(best_pred_id) in pred_map:
        target_box_min, target_box_max = _bbox_from_record(pred_map[int(best_pred_id)])
        _draw_bbox_3d(ax, target_box_min, target_box_max, color=COLOR_TARGET, linewidth=2.4, alpha=1.0)

    if gt_record is not None:
        gt_box_min, gt_box_max = _bbox_from_record(gt_record)
        _draw_bbox_3d(ax, gt_box_min, gt_box_max, color=COLOR_GT, linewidth=1.6, alpha=0.95)

    x0, y0, z0 = map(float, room_min)
    x1, y1, z1 = map(float, room_max)
    pad_x = max((x1 - x0) * 0.04, 0.08)
    pad_y = max((y1 - y0) * 0.04, 0.08)
    pad_z = max((z1 - z0) * 0.04, 0.08)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_zlim(z0 - pad_z, z1 + pad_z)
    ax.set_title("3D point cloud + instance masks", fontsize=FONT_PANEL, loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-58)
    try:
        ax.set_box_aspect((x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y, z1 - z0 + 2 * pad_z))
    except AttributeError:
        pass
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.grid(alpha=0.10, linestyle=":", linewidth=0.7)


def _short_rel_summary(result: Mapping[str, Any]) -> str:
    rels = result.get("relation_scores") or {}
    if not rels:
        return "No relation score"
    parts = []
    for payload in rels.values():
        parts.append(f"{payload.get('type')}={_safe_float(payload.get('score')):.2f}")
    return ", ".join(parts[:3])


def _draw_topk_panel(ax: plt.Axes, record: Mapping[str, Any], gt_record: Optional[Mapping[str, Any]], max_rank: int) -> None:
    ax.set_title("Top-k candidates", fontsize=FONT_PANEL, loc="left")
    top_results = list(record.get("graph_top_results") or [])[:max_rank]
    if not top_results:
        ax.axis("off")
        ax.text(0.5, 0.5, "No ranking", ha="center", va="center")
        return
    gt_center = _center_from_record(gt_record) if gt_record is not None else None
    labels, scores, colors = [], [], []
    for rank, item in enumerate(top_results, 1):
        pred_id = item.get("pred_id")
        cls = _shorten(item.get("class_name") or "object", 12)
        scores.append(_safe_float(item.get("total_score")))
        note = cls
        if gt_center is not None and isinstance(item.get("center"), (list, tuple)):
            dist = float(np.linalg.norm(np.asarray(item["center"], dtype=np.float64) - gt_center))
            note += f" d={dist:.2f}"
        labels.append(f"{rank}. #{pred_id} {note}")
        colors.append(COLOR_TOPK[min(rank - 1, len(COLOR_TOPK) - 1)])
    y = np.arange(len(labels))
    ax.barh(y, scores, color=colors, alpha=0.9, edgecolor="#555555", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_BODY)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(1.0, max(scores) * 1.10))
    ax.set_xlabel("score")
    ax.grid(axis="x", linestyle=":", alpha=0.16)
    ax.tick_params(axis="y", pad=2)
    ax.margins(y=0.10)


def _flatten_breakdown(result: Mapping[str, Any]) -> List[Tuple[str, float]]:
    rows = []
    for node_id, payload in (result.get("node_scores") or {}).items():
        for key, value in (payload.get("components") or {}).items():
            rows.append((f"{node_id}.{key}", _safe_float(value)))
    for payload in (result.get("relation_scores") or {}).values():
        rel = str(payload.get("type") or "rel")
        rows.append((rel, _safe_float(payload.get("score"))))
    rows.sort(key=lambda item: item[0])
    return rows[:9]


def _draw_breakdown_panel(ax: plt.Axes, record: Mapping[str, Any]) -> None:
    ax.set_title("Score breakdown", fontsize=FONT_PANEL, loc="left")
    top_results = record.get("graph_top_results") or []
    if not top_results:
        ax.axis("off")
        ax.text(0.5, 0.5, "No breakdown", ha="center", va="center")
        return
    rows = _flatten_breakdown(top_results[0])
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No breakdown", ha="center", va="center")
        return
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    y = np.arange(len(rows))
    ax.barh(y, values, color="#8D99AE", alpha=0.9, edgecolor="#555555", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_SMALL)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("component score")
    ax.grid(axis="x", linestyle=":", alpha=0.16)
    for idx, value in enumerate(values):
        ax.text(min(value + 0.015, 1.01), idx, f"{value:.3f}", va="center", fontsize=8.5, clip_on=True)
    ax.tick_params(axis="y", pad=2)
    ax.margins(y=0.10)


def _draw_header(ax: plt.Axes, record: Mapping[str, Any], scene_data: Mapping[str, Any], gt_record: Optional[Mapping[str, Any]]) -> None:
    ax.axis("off")
    scene_id = record.get("scene_id")
    ref_id = record.get("ref_id")
    best_pred_id = record.get("best_pred_id")
    distance = record.get("distance")
    desc = str(record.get("description") or "").strip()
    top_results = record.get("graph_top_results") or []
    top1 = top_results[0] if top_results else {}
    rel_summary = _short_rel_summary(top1)

    title = f"{scene_id} | ref_id={ref_id} | pred={best_pred_id}"
    if isinstance(distance, (int, float)):
        title += f" | distance={distance:.3f}"

    data_note = (
        "Point cloud and masks are real scene data from "
        f"{scene_data['scene_dir'] / 'final.ply'} and {scene_data['scene_dir'] / 'ckpt_final.npz'}; "
        "only colors, alpha, and GT bbox overlay are visualization choices."
    )
    score_note = (
        "Top-k comes directly from graph_top_results in the query-graph output; "
        "score breakdown comes from node_scores and relation_scores of the top-1 result."
    )
    desc_block = _wrap(f"\"{desc}\"", 105)
    note_block = _wrap(data_note, 118)
    score_block = _wrap(score_note, 118)

    ax.text(0.00, 0.95, title, fontsize=FONT_HEADER, fontweight="bold", ha="left", va="top")
    ax.text(0.00, 0.60, desc_block, fontsize=FONT_BODY, ha="left", va="top")
    ax.text(0.00, 0.18, f"Top-1 relation summary: {rel_summary}", fontsize=FONT_SMALL, ha="left", va="top")
    ax.text(0.52, 0.95, note_block, fontsize=FONT_HEADER_BODY, ha="left", va="top", color="#444444")
    ax.text(0.52, 0.40, score_block, fontsize=FONT_HEADER_BODY, ha="left", va="top", color="#444444")


def _scene_graph_edges(scene_graph: SpatialSceneGraph, top_k_neighbors: int = 2) -> List[Tuple[int, int, str]]:
    edges: List[Tuple[int, int, str]] = []
    objects = list(scene_graph.objects)
    if not objects:
        return edges
    for src in objects:
        candidates: List[Tuple[float, int, str]] = []
        for dst in objects:
            if src.pred_id == dst.pred_id:
                continue
            edge = scene_graph.object_edge(src.pred_id, dst.pred_id)
            dist_xy = float(edge["distance_xy"])
            dz = float(edge["dz"])
            relation = "near"
            if abs(dz) > scene_graph.vertical_scale * 0.6:
                relation = "above" if dz > 0 else "below"
            candidates.append((dist_xy, dst.pred_id, relation))
        candidates.sort(key=lambda item: item[0])
        for _, dst_id, relation in candidates[:top_k_neighbors]:
            edges.append((src.pred_id, dst_id, relation))
    dedup: Dict[Tuple[int, int], str] = {}
    for src_id, dst_id, relation in edges:
        key = (src_id, dst_id)
        if key not in dedup:
            dedup[key] = relation
    return [(src_id, dst_id, rel) for (src_id, dst_id), rel in dedup.items()]


def _draw_scene_graph_figure(
    record: Mapping[str, Any],
    *,
    scene_data: Mapping[str, Any],
    out_path: Path,
    dpi: int,
) -> Dict[str, Any]:
    _set_style()
    fig = plt.figure(figsize=(12.5, 10.0))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.18, 0.82], hspace=0.10)
    ax_head = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_head.axis("off")

    scene_id = record.get("scene_id")
    ref_id = record.get("ref_id")
    best_pred_id = record.get("best_pred_id")
    scene_graph = scene_data["scene_graph"]
    objects = list(scene_graph.objects)
    room_min = scene_data["room_min"]
    room_max = scene_data["room_max"]
    best_assignments = {}
    top_results = record.get("graph_top_results") or []
    if top_results:
        best_assignments = top_results[0].get("assignments") or {}
    ref_ids = {int(payload["pred_id"]) for node_id, payload in best_assignments.items() if node_id != "t" and payload.get("pred_id") is not None}
    highlight_ids = set(ref_ids)
    if best_pred_id is not None:
        highlight_ids.add(int(best_pred_id))

    ax_head.text(
        0.0,
        0.92,
        f"Scene graph | {scene_id} | ref_id={ref_id} | pred={best_pred_id}",
        fontsize=FONT_HEADER,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_head.text(
        0.0,
        0.45,
        "This graph is derived from real prediction instances in pred_bboxes.json and the SpatialSceneGraph relations built from object centers and box geometry.",
        fontsize=FONT_BODY,
        ha="left",
        va="top",
        color="#444444",
    )

    x0 = float(room_min[0])
    x1 = float(room_max[0])
    y0 = float(room_min[1])
    y1 = float(room_max[1])
    pad_x = max((x1 - x0) * 0.04, 0.08)
    pad_y = max((y1 - y0) * 0.04, 0.08)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scene graph layout in XY plane", fontsize=FONT_PANEL, loc="left")
    ax.grid(alpha=0.10, linestyle=":", linewidth=0.7)

    edges = _scene_graph_edges(scene_graph, top_k_neighbors=2)
    centers = {obj.pred_id: obj.center for obj in objects}
    room_scale = max(float(np.linalg.norm((room_max - room_min)[:2])), 1e-6)
    label_payload = {"edge_labels": [], "node_labels": []}
    for src_id, dst_id, relation in edges:
        a = centers[src_id]
        b = centers[dst_id]
        is_highlight_edge = src_id in highlight_ids or dst_id in highlight_ids
        edge_color = "#7D7D7D" if is_highlight_edge else "#BEBEBE"
        edge_alpha = 0.72 if is_highlight_edge else 0.35
        edge_width = 2.2 if is_highlight_edge else 1.35
        ax.plot([a[0], b[0]], [a[1], b[1]], color=edge_color, alpha=edge_alpha, linewidth=edge_width, zorder=1)
        if is_highlight_edge:
            mx = (a[0] + b[0]) / 2.0
            my = (a[1] + b[1]) / 2.0
            dx = float(b[0] - a[0])
            dy = float(b[1] - a[1])
            norm = max(math.hypot(dx, dy), 1e-6)
            off = 0.018 * room_scale
            ox = -dy / norm * off
            oy = dx / norm * off
            sign = -1.0 if ((src_id + dst_id) % 2) else 1.0
            tx = mx + sign * ox
            ty = my + sign * oy
            ax.text(
                tx,
                ty,
                relation,
                fontsize=FONT_SMALL,
                color="#555555",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.14", facecolor=COLOR_BG, edgecolor="#A0A0A0", linewidth=0.7, alpha=0.96),
            )
            label_payload["edge_labels"].append(
                {
                    "src_id": int(src_id),
                    "dst_id": int(dst_id),
                    "relation": relation,
                    "x": float(tx),
                    "y": float(ty),
                }
            )

    for obj in objects:
        pred_id = int(obj.pred_id)
        x = float(obj.center[0])
        y = float(obj.center[1])
        if pred_id == best_pred_id:
            color = COLOR_TARGET
            size = 180
            z = 4
        elif pred_id in ref_ids:
            color = COLOR_REF
            size = 138
            z = 3
        else:
            color = "#6E6E6E"
            size = 64
            z = 2
        ax.scatter([x], [y], s=size, c=color, alpha=0.95, edgecolors="white", linewidths=0.9, zorder=z)
        if pred_id == best_pred_id or pred_id in ref_ids:
            label_text = f"{obj.class_name}\n#{pred_id}"
            ax.text(
                x,
                y,
                label_text,
                fontsize=FONT_SMALL,
                ha="center",
                va="bottom",
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=COLOR_BG, edgecolor=color, linewidth=0.9, alpha=0.97),
                zorder=5,
            )
            label_payload["node_labels"].append(
                {
                    "pred_id": pred_id,
                    "class_name": obj.class_name,
                    "text": label_text,
                    "x": x,
                    "y": y,
                }
            )

    legend_items = [
        Patch(facecolor=COLOR_TARGET, edgecolor="none", label="predicted target"),
        Patch(facecolor=COLOR_REF, edgecolor="none", label="reference objects"),
        Patch(facecolor="#8F8F8F", edgecolor="none", label="other scene objects"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=FONT_BODY)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return label_payload


def _render_pointcloud_panel_figure(
    record: Mapping[str, Any],
    *,
    scene_data: Mapping[str, Any],
    gt_record: Optional[Mapping[str, Any]],
    plane: str,
    out_path: Path,
    max_rank: int,
    dpi: int,
    rasterize_points: bool,
) -> None:
    _set_style()
    figsize = (10.5, 9.0) if plane == "xy" else (10.5, 7.2)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    _draw_pointcloud_panel(
        ax,
        scene_data=scene_data,
        record=record,
        gt_record=gt_record,
        plane=plane,
        max_rank=max_rank,
        rasterize_points=rasterize_points,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_pointcloud_panel_3d_figure(
    record: Mapping[str, Any],
    *,
    scene_data: Mapping[str, Any],
    gt_record: Optional[Mapping[str, Any]],
    out_path: Path,
    max_rank: int,
    dpi: int,
    rasterize_points: bool,
) -> None:
    _set_style()
    fig = plt.figure(figsize=(11.0, 8.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _draw_pointcloud_panel_3d(
        ax,
        scene_data=scene_data,
        record=record,
        gt_record=gt_record,
        max_rank=max_rank,
        rasterize_points=rasterize_points,
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_query_graph_figure(record: Mapping[str, Any], *, out_path: Path, dpi: int) -> None:
    _set_style()
    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(1, 1, 1)
    _draw_query_graph(ax, record.get("query_graph") or {})
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_ranking_figure(
    record: Mapping[str, Any],
    *,
    gt_record: Optional[Mapping[str, Any]],
    out_path: Path,
    max_rank: int,
    dpi: int,
) -> None:
    _set_style()
    fig = plt.figure(figsize=(11.0, 8.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.54, 0.46], hspace=0.34)
    ax_topk = fig.add_subplot(gs[0, 0])
    ax_breakdown = fig.add_subplot(gs[1, 0])
    _draw_topk_panel(ax_topk, record, gt_record, max_rank=max_rank)
    _draw_breakdown_panel(ax_breakdown, record)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.93, bottom=0.10, hspace=0.38)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_panel_exports(
    record: Mapping[str, Any],
    *,
    scene_data: Mapping[str, Any],
    gt_record: Optional[Mapping[str, Any]],
    outdir: Path,
    stem: str,
    max_rank: int,
    dpi: int,
) -> Dict[str, Dict[str, str]]:
    panel_paths: Dict[str, Dict[str, str]] = {}
    panel_specs = [
        ("xy", outdir / f"{stem}_xy.png", outdir / f"{stem}_xy.svg"),
        ("xz", outdir / f"{stem}_xz.png", outdir / f"{stem}_xz.svg"),
    ]
    for plane, png_path, svg_path in panel_specs:
        _render_pointcloud_panel_figure(
            record,
            scene_data=scene_data,
            gt_record=gt_record,
            plane=plane,
            out_path=png_path,
            max_rank=max_rank,
            dpi=dpi,
            rasterize_points=True,
        )
        _render_pointcloud_panel_figure(
            record,
            scene_data=scene_data,
            gt_record=gt_record,
            plane=plane,
            out_path=svg_path,
            max_rank=max_rank,
            dpi=dpi,
            rasterize_points=False,
        )
        panel_paths[plane] = {"png": str(png_path), "svg": str(svg_path)}

    point3d_png_path = outdir / f"{stem}_3d.png"
    point3d_svg_path = outdir / f"{stem}_3d.svg"
    _render_pointcloud_panel_3d_figure(
        record,
        scene_data=scene_data,
        gt_record=gt_record,
        out_path=point3d_png_path,
        max_rank=max_rank,
        dpi=dpi,
        rasterize_points=True,
    )
    _render_pointcloud_panel_3d_figure(
        record,
        scene_data=scene_data,
        gt_record=gt_record,
        out_path=point3d_svg_path,
        max_rank=max_rank,
        dpi=dpi,
        rasterize_points=False,
    )
    panel_paths["pointcloud_3d"] = {"png": str(point3d_png_path), "svg": str(point3d_svg_path)}

    query_png_path = outdir / f"{stem}_query_graph.png"
    query_svg_path = outdir / f"{stem}_query_graph.svg"
    _render_query_graph_figure(record, out_path=query_png_path, dpi=dpi)
    _render_query_graph_figure(record, out_path=query_svg_path, dpi=dpi)
    panel_paths["query_graph"] = {"png": str(query_png_path), "svg": str(query_svg_path)}

    ranking_png_path = outdir / f"{stem}_ranking.png"
    ranking_svg_path = outdir / f"{stem}_ranking.svg"
    _render_ranking_figure(record, gt_record=gt_record, out_path=ranking_png_path, max_rank=max_rank, dpi=dpi)
    _render_ranking_figure(record, gt_record=gt_record, out_path=ranking_svg_path, max_rank=max_rank, dpi=dpi)
    panel_paths["ranking"] = {"png": str(ranking_png_path), "svg": str(ranking_svg_path)}

    return panel_paths


def _render_single_figure(
    record: Mapping[str, Any],
    *,
    scene_data: Mapping[str, Any],
    gt_record: Optional[Mapping[str, Any]],
    out_path: Path,
    max_rank: int,
    dpi: int,
) -> Dict[str, Any]:
    _set_style()
    fig = plt.figure(figsize=(17.5, 10.8))
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.22, 0.48, 0.30],
        width_ratios=[1.15, 1.15, 0.92],
        hspace=0.24,
        wspace=0.22,
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_xy = fig.add_subplot(gs[1, 0:2])
    ax_xz = fig.add_subplot(gs[2, 0:2])
    ax_graph = fig.add_subplot(gs[1, 2])
    sub_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 2], hspace=0.52)
    ax_topk = fig.add_subplot(sub_right[0, 0])
    ax_breakdown = fig.add_subplot(sub_right[1, 0])

    _draw_header(ax_header, record, scene_data, gt_record)
    _draw_pointcloud_panel(ax_xy, scene_data=scene_data, record=record, gt_record=gt_record, plane="xy", max_rank=max_rank)
    _draw_pointcloud_panel(ax_xz, scene_data=scene_data, record=record, gt_record=gt_record, plane="xz", max_rank=max_rank)
    _draw_query_graph(ax_graph, record.get("query_graph") or {})
    _draw_topk_panel(ax_topk, record, gt_record, max_rank=max_rank)
    _draw_breakdown_panel(ax_breakdown, record)

    legend_items = [
        Patch(facecolor=COLOR_TOPK[0], edgecolor="none", label="predicted target mask"),
        Patch(facecolor=COLOR_REF, edgecolor="none", label="reference masks"),
        Patch(facecolor="none", edgecolor=COLOR_GT, linestyle=(0, (4, 2)), label="GT box"),
        Patch(facecolor="#B0B0B0", edgecolor="none", label="scene point cloud"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=FONT_BODY,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "scene_id": record.get("scene_id"),
        "ref_id": record.get("ref_id"),
        "best_pred_id": record.get("best_pred_id"),
        "distance": record.get("distance"),
        "description": record.get("description"),
        "figure_path": str(out_path),
        "data_source": {
            "point_cloud": str(scene_data["scene_dir"] / "final.ply"),
            "instance_masks": str(scene_data["scene_dir"] / "ckpt_final.npz"),
            "pred_bboxes": str(scene_data["scene_dir"] / "pred_bboxes.json"),
        },
        "score_source": {
            "top_k": "graph_top_results from the query-graph grounding output JSONL",
            "breakdown": "top-1 node_scores and relation_scores from graph_top_results[0]",
        },
        "what_is_real": "The point cloud coordinates and instance masks come directly from final.ply and ckpt_final.npz. The colored overlays, transparency, ranking colors, and GT dashed rectangle are visualization layers added by this script.",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one large figure per successful NR3D example.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--top-n", type=int, default=4, help="How many successful examples to render")
    parser.add_argument("--max-rank", type=int, default=5, help="How many top-k candidates to display")
    parser.add_argument("--success-distance", type=float, default=SUCCESS_DISTANCE_THRESH)
    parser.add_argument("--ref-id", type=int, action="append", default=None)
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results_path = args.results.resolve()
    gt_path = args.gt.resolve()
    scene_root = args.scene_root.resolve()
    outdir = args.outdir.resolve()

    if not results_path.is_file():
        raise SystemExit(f"Results file not found: {results_path}")
    if not gt_path.is_file():
        raise SystemExit(f"GT file not found: {gt_path}")
    if not scene_root.is_dir():
        raise SystemExit(f"Scene root not found: {scene_root}")

    records = _load_jsonl(results_path)
    gt_index = _build_gt_index(gt_path)
    selected = _select_records(
        records,
        scene_root=scene_root,
        success_distance=args.success_distance,
        top_n=args.top_n,
        ref_ids=args.ref_id or [],
        scene_ids=args.scene_id or [],
    )
    if not selected:
        raise SystemExit("No successful examples with full scene assets were found")

    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    scene_cache: Dict[str, Dict[str, Any]] = {}
    for record in selected:
        scene_id = str(record.get("scene_id") or "")
        if scene_id not in scene_cache:
            scene_cache[scene_id] = _load_scene_data(scene_root, scene_id)
        scene_data = scene_cache[scene_id]
        gt_record = _lookup_gt(record, gt_index)
        stem = _sanitize_slug(f"{scene_id}_ref{record.get('ref_id')}_pred{record.get('best_pred_id')}")
        out_path = outdir / f"{stem}.png"
        scene_graph_path = outdir / f"{stem}_scene_graph.png"
        scene_graph_svg_path = outdir / f"{stem}_scene_graph.svg"
        scene_graph_labels_path = outdir / f"{stem}_scene_graph_labels.json"
        entry = _render_single_figure(
            record,
            scene_data=scene_data,
            gt_record=gt_record,
            out_path=out_path,
            max_rank=args.max_rank,
            dpi=args.dpi,
        )
        label_payload = _draw_scene_graph_figure(
            record,
            scene_data=scene_data,
            out_path=scene_graph_path,
            dpi=args.dpi,
        )
        _draw_scene_graph_figure(
            record,
            scene_data=scene_data,
            out_path=scene_graph_svg_path,
            dpi=args.dpi,
        )
        panel_exports = _render_panel_exports(
            record,
            scene_data=scene_data,
            gt_record=gt_record,
            outdir=outdir,
            stem=stem,
            max_rank=args.max_rank,
            dpi=args.dpi,
        )
        scene_graph_labels_path.write_text(json.dumps(label_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        entry["scene_graph_figure_path"] = str(scene_graph_path)
        entry["scene_graph_svg_path"] = str(scene_graph_svg_path)
        entry["scene_graph_labels_path"] = str(scene_graph_labels_path)
        entry["panel_exports"] = panel_exports
        manifest.append(entry)
        print(f"Saved {out_path}")
        print(f"Saved {scene_graph_path}")
        print(f"Saved {scene_graph_svg_path}")
        print(f"Saved {scene_graph_labels_path}")
        for panel_name, payload in panel_exports.items():
            print(f"Saved {payload['png']}")
            print(f"Saved {payload['svg']}")

    manifest_path = outdir / "single_fig_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
