#!/usr/bin/env python
"""Match NR3D GT records to predicted bboxes by center proximity and semantic label.

Outputs a JSON list of matched GT records and prints match ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


DEFAULT_SYNONYM_GROUPS = [
    {"sofa", "couch", "loveseat"},
    {"tv", "television", "monitor", "tv screen"},
    {"cabinet", "cabinets", "kitchen cabinet", "kitchen cabinets", "cupboard"},
    {"counter", "countertop", "kitchen counter"},
    {"trash can", "garbage can", "trash bin", "garbage bin", "waste bin"},
    {"refrigerator", "fridge"},
    {"nightstand", "bedside table", "bedside cabinet"},
    {"coffee table", "cocktail table"},
    {"tv stand", "television stand", "media stand"},
    {"picture", "painting", "poster"},
    {"stove", "oven", "range"},
]


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalize_label(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [singularize_token(tok) for tok in text.split()]
    return " ".join(tokens)


def load_synonym_groups(extra_path: Optional[Path]) -> List[Set[str]]:
    groups: List[Set[str]] = []
    for g in DEFAULT_SYNONYM_GROUPS:
        groups.append({normalize_label(x) for x in g})

    if not extra_path:
        return groups

    if not extra_path.exists():
        raise FileNotFoundError(f"Synonyms file not found: {extra_path}")

    with extra_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        for k, v in data.items():
            items = [k] + (v if isinstance(v, list) else [v])
            groups.append({normalize_label(x) for x in items})
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                groups.append({normalize_label(x) for x in item})
            elif isinstance(item, dict):
                for k, v in item.items():
                    items = [k] + (v if isinstance(v, list) else [v])
                    groups.append({normalize_label(x) for x in items})
            elif isinstance(item, str):
                groups.append({normalize_label(item)})
    else:
        raise ValueError("Unsupported synonyms file format. Use dict or list.")

    return groups


def semantic_match(a: str, b: str, groups: Sequence[Set[str]]) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True

    for group in groups:
        if a in group and b in group:
            return True

    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return False

    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if a_set.issubset(b_set) or b_set.issubset(a_set):
        return True

    return False


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def load_pred_for_scene(pred_dir: Path, scene_id: str) -> Optional[List[dict]]:
    path = pred_dir / scene_id / "pred_bboxes.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match NR3D GT records to predicted bboxes; output matched GT records."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("sr3d/sr3d_gt_bboxes.json"),
        help="Path to nr3d_gt_bboxes.json",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("output_test/nr3d"),
        help="Directory containing per-scene pred_bboxes.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sr3d/sr3d_gt_bboxes_matched.json"),
        help="Output JSON path for matched GT records",
    )
    parser.add_argument(
        "--dist-thresh",
        type=float,
        default=0.3,
        help="Center distance threshold (meters) for matching",
    )
    parser.add_argument(
        "--synonyms",
        type=Path,
        default=None,
        help="Optional JSON file to extend synonym groups",
    )

    args = parser.parse_args()

    with args.gt.open() as f:
        gt_data = json.load(f)

    if not isinstance(gt_data, list):
        raise ValueError("GT file should be a list of records")

    groups = load_synonym_groups(args.synonyms)

    # Group GT by scene to avoid loading all pred files at once.
    gt_by_scene: Dict[str, List[dict]] = {}
    for rec in gt_data:
        scene_id = rec.get("scene_id")
        if not scene_id:
            continue
        gt_by_scene.setdefault(scene_id, []).append(rec)

    matched: List[dict] = []
    missing_pred_scenes: List[str] = []

    total = len(gt_data)
    matched_count = 0

    for scene_id, gt_list in gt_by_scene.items():
        preds = load_pred_for_scene(args.pred_dir, scene_id)
        if preds is None:
            missing_pred_scenes.append(scene_id)
            continue
        pred_label_norms = [normalize_label(p.get("class_name", "")) for p in preds]

        for gt in gt_list:
            gt_center = gt.get("center")
            if not gt_center or len(gt_center) != 3:
                continue
            gt_label_norm = normalize_label(gt.get("label", ""))
            if not gt_label_norm:
                continue

            # Find any pred that matches by semantic + center distance.
            best_dist = None
            matched_pred = None
            for pred, pred_label_norm in zip(preds, pred_label_norms):
                if not semantic_match(gt_label_norm, pred_label_norm, groups):
                    continue
                pred_center = pred.get("center")
                if not pred_center or len(pred_center) != 3:
                    continue
                dist = euclidean(gt_center, pred_center)
                if dist > args.dist_thresh:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    matched_pred = pred

            if matched_pred is not None:
                matched.append(gt)
                matched_count += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(matched, f, indent=2)

    ratio = matched_count / total if total else 0.0
    print(f"Total GT: {total}")
    print(f"Matched: {matched_count}")
    print(f"Match ratio: {ratio:.4f}")
    if missing_pred_scenes:
        print(f"Missing pred_bboxes.json for {len(missing_pred_scenes)} scenes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
