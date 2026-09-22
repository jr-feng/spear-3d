#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def iter_scene_dirs(scans_root: Path):
    for scene_dir in sorted(scans_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        agg_file = scene_dir / f"{scene_id}_vh_clean.aggregation.json"
        yield scene_id, agg_file


def extract_labels(agg_file: Path, include_background: bool, background_label: str):
    with agg_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    seen = set()
    for group in data.get("segGroups", []):
        label = group.get("label")
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)

    if include_background and background_label not in seen:
        labels.insert(0, background_label)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract ordered object labels from ScanNet aggregation JSON files."
    )
    parser.add_argument(
        "--scans-root",
        default="/workspace/nr3d/scans",
        help="Path to ScanNet scans root (default: /workspace/nr3d/scans).",
    )
    parser.add_argument(
        "--output",
        default="./nr3d_label.json",
        help="Output JSON path. If empty, print to stdout.",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Disable adding a background label at the start of each TEXT_PROMPT list.",
    )
    parser.add_argument(
        "--background-label",
        default="background",
        help="Background label string (default: background).",
    )
    args = parser.parse_args()

    scans_root = Path(args.scans_root)
    if not scans_root.exists():
        print(f"[error] scans root not found: {scans_root}", file=sys.stderr)
        return 1

    results = []
    for scene_id, agg_file in iter_scene_dirs(scans_root):
        if not agg_file.exists():
            print(f"[warn] missing aggregation file: {agg_file}", file=sys.stderr)
            continue
        include_background = not args.no_background
        labels = extract_labels(agg_file, include_background, args.background_label)
        results.append({"sceneId": scene_id, "TEXT_PROMPT": labels})

    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_json + "\n", encoding="utf-8")
    else:
        print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
