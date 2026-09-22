import argparse
import json
import random
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on runtime environment
    OpenAI = None

from spatial_scene_graph import (
    SceneGraphQueryEngine,
    SpatialSceneGraph,
    compile_query_from_parsed_chain,
)


# 可调参数
OAS_ROOT = Path("output_test/nr3d")
#GT_PATH = Path("./nr3d/nr3d_gt_bboxes_matched.json")
GT_PATH = Path("./sr3d/sr3d_pred_bboxes_matched.json")
PRED_NAME = "pred_bboxes.json"
MAX_GT = 500
PROGRESS_BAR_WIDTH = 30
DIST_THRESH_30 = 0.3
DIST_THRESH_50 = 0.5


def make_client():
    if OpenAI is None:
        raise RuntimeError("The `openai` package is required to run this script.")
    return OpenAI(
        base_url="https://api.silra.cn/v1",
        api_key="sk-CvP9I6FP8Kzv06XzCYdu1WnLQmCNbVuTosIWZqVdJcAEI2Ib",
    )


def build_prompt(gt_rec, preds):
    scene = gt_rec["scene_id"]
    desc = gt_rec.get("description") or gt_rec.get("gt_description") or ""

    class_counts = {}
    for pred in preds:
        cls = str(pred.get("class_name") or "object").strip()
        class_counts[cls] = class_counts.get(cls, 0) + 1
    class_inventory = [
        {"class_name": class_name, "count": count}
        for class_name, count in sorted(class_counts.items(), key=lambda item: (item[0], item[1]))
    ]

    example_section = """
Example:
Description: "the chair to the right of the trash can."
Example output:
{
  "scene_id":"scene0000_00",
  "best_direction":"+x",
  "parser_confidence":0.92,
  "skip_due_to_size":false,
  "reason":"chair is constrained by a right-of relation to trash can",
  "parsed_chain":[
    {"role":"ref","class":"trash can","constraints":[]},
    {"role":"target","class":"chair","constraints":["right of ref"]}
  ]
}

Example:
Description: "the middle of three upper cabinets to the right of the refrigerator."
Example output:
{
  "scene_id":"scene0000_00",
  "best_direction":"+x",
  "parser_confidence":0.89,
  "skip_due_to_size":false,
  "reason":"cabinet constrained by refrigerator, upper position, and middle-of-group relation",
  "parsed_chain":[
    {"role":"ref","class":"refrigerator","constraints":[]},
    {"role":"target","class":"cabinet","constraints":["upper","right of ref","middle of group of three"]}
  ]
}

Example:
Description: "the largest chair near the table."
Example output:
{
  "scene_id":"scene0000_00",
  "best_direction":null,
  "parser_confidence":0.97,
  "skip_due_to_size":true,
  "reason":"target is mainly distinguished by size",
  "parsed_chain":[
    {"role":"ref","class":"table","constraints":[]},
    {"role":"target","class":"chair","constraints":["largest","near ref"]}
  ]
}
"""

    prompt = f"""You are a 3D referring-expression parser for a downstream spatial scene graph reasoner.
Your job is ONLY to convert the language description into a compact structured query.
Do NOT choose a predicted object id. Do NOT do detailed geometric calculations.

Available candidate class inventory for this scene:
{json.dumps(class_inventory, ensure_ascii=False, indent=2)}

World setup:
- The scene uses a fixed 3D world coordinate system (x, y, z).
- Words like front/back/left/right are viewer-centric.
- If the description clearly implies a viewing direction, set best_direction to one of "+x", "-x", "+y", "-y".
- If the direction is not needed or cannot be inferred reliably, set best_direction to null.

Output schema:
{{
  "scene_id": "{scene}",
  "best_direction": "+x/-x/+y/-y or null",
  "parser_confidence": 0-1,
  "skip_due_to_size": true/false,
  "reason": "...",
  "parsed_chain": [
    {{"role": "ref", "class": "...", "constraints": []}},
    {{"role": "target", "class": "...", "constraints": ["..."]}}
  ]
}}

Rules:
1) parsed_chain must contain exactly one target node.
2) Add zero, one, or multiple ref nodes if needed, in mention order.
3) Prefer class names that match the candidate class inventory by synonym or singular/plural form.
4) The target constraints should use concise phrases from this style:
   - "left of ref", "right of ref", "in front of ref", "behind ref"
   - "near ref", "closest to ref", "farthest from ref"
   - "between ref1 and ref2"
   - "upper", "lower", "leftmost", "rightmost", "frontmost", "backmost"
   - "on top of ref", "above ref", "under ref", "below ref"
   - "against wall", "on the wall", "corner", "center of room"
   - "side of room with ref"
   - "with ref", "with ref on it"
   - "group of two", "group of three", "middle of group of three", "end of row"
5) If the target is mainly distinguished by size words such as largest/smallest/big/small/tall/short,
   set skip_due_to_size=true.
6) Keep reason concise, at most 2 short sentences.
7) Output JSON only, no extra text.

Target description:
"{desc}"

{example_section}
"""
    return prompt


def _normalize_vec3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _bbox_from_center_size(center, size):
    center = _normalize_vec3(center)
    size = _normalize_vec3(size)
    if center is None or size is None:
        return None
    return (
        [c - s / 2.0 for c, s in zip(center, size)],
        [c + s / 2.0 for c, s in zip(center, size)],
    )


def _bbox_from_min_max(box_min, box_max):
    box_min = _normalize_vec3(box_min)
    box_max = _normalize_vec3(box_max)
    if box_min is None or box_max is None:
        return None
    return box_min, box_max


def _center_from_box(box):
    if box is None:
        return None
    b_min, b_max = box
    return [(b_min[i] + b_max[i]) / 2.0 for i in range(3)]


def _normalize_pred_id(pred_id):
    if pred_id is None:
        return None
    try:
        return int(pred_id)
    except (TypeError, ValueError):
        return None


def _pred_center_from_record(pred_rec):
    if not pred_rec:
        return None
    center = _normalize_vec3(pred_rec.get("center"))
    if center is not None:
        return center
    box = _bbox_from_min_max(pred_rec.get("box_min"), pred_rec.get("box_max"))
    if box is not None:
        return _center_from_box(box)
    box = _bbox_from_center_size(pred_rec.get("center"), pred_rec.get("size"))
    return _center_from_box(box)


def _compute_distance_for_prediction(gt_rec, pred_map, pred_id):
    gt_center = _normalize_vec3(gt_rec.get("center"))
    pred_rec = pred_map.get(pred_id) if pred_id is not None else None
    pred_center = _pred_center_from_record(pred_rec)
    if gt_center is None or pred_center is None:
        return None
    dx = gt_center[0] - pred_center[0]
    dy = gt_center[1] - pred_center[1]
    dz = gt_center[2] - pred_center[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _summarize_dists(dists):
    if not dists:
        return None
    valid = [d for d in dists if d is not None]
    valid_total = len(valid)
    total = len(dists)
    if valid_total == 0:
        return None, None, None, 0, total
    mean_dist = sum(valid) / valid_total
    acc30 = sum(1 for d in valid if d <= DIST_THRESH_30) / valid_total
    acc50 = sum(1 for d in valid if d <= DIST_THRESH_50) / valid_total
    return mean_dist, acc30, acc50, valid_total, total


def _render_progress(current, total, processed, skipped, width=PROGRESS_BAR_WIDTH):
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} processed={processed} skipped={skipped}"


def _make_progress_printer(width=PROGRESS_BAR_WIDTH):
    last_len = 0

    def _print_progress(current, total, processed, skipped):
        nonlocal last_len
        msg = _render_progress(current, total, processed, skipped, width=width)
        pad = " " * max(0, last_len - len(msg))
        print("\r" + msg + pad, end="", flush=True)
        last_len = len(msg)

    return _print_progress


def parse_response_json(text: str):
    text = (text or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _extract_graph_prediction(parsed, engine, top_k):
    if not isinstance(parsed, dict):
        return None, None, "parsed response is not a JSON object"

    if parsed.get("skip_due_to_size"):
        return None, None, None

    parsed_chain = parsed.get("parsed_chain")
    if not isinstance(parsed_chain, list) or not parsed_chain:
        return None, None, "parsed_chain is missing or empty"

    viewer_direction = parsed.get("best_direction")
    if viewer_direction in {"null", "", "none"}:
        viewer_direction = None

    try:
        graph_query = compile_query_from_parsed_chain(
            parsed_chain,
            viewer_direction=viewer_direction,
            fallback_to_all=True,
        )
        graph_result = engine.score_query(graph_query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)

    results = graph_result.get("results") or []
    if not results:
        return None, graph_result, None
    return results[0].get("pred_id"), graph_result, None


def main():
    ap = argparse.ArgumentParser(description="Run LLM parser + spatial scene graph grounding and write results to JSONL.")
    ap.add_argument("--scene", default=None, help="Only process a single scene_id")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max number of GT descriptions to process (default: 500, cap at 500)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output_test/nr3d/llm_graph_outputs.json"),
        help="Output JSONL path",
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("output_test/nr3d/llm_graph_metrics.json"),
        help="Output metrics JSON path",
    )
    ap.add_argument("--print-every", type=int, default=50, help="Progress print interval (0 to disable)")
    ap.add_argument("--top-k", type=int, default=5, help="How many graph-ranked candidates to save")
    args = ap.parse_args()

    gts = json.loads(GT_PATH.read_text())
    if args.scene:
        gts = [rec for rec in gts if rec.get("scene_id") == args.scene]

    random.shuffle(gts)
    max_limit = MAX_GT if args.max is None else min(args.max, MAX_GT)
    if max_limit is not None:
        gts = gts[:max_limit]

    if not gts:
        raise SystemExit("没有匹配到可处理的 GT 描述记录")

    pred_cache = {}
    pred_map_cache = {}
    graph_engine_cache = {}
    processed = 0
    skipped = 0
    total = len(gts)
    dists = []
    progress = _make_progress_printer()
    client = make_client()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f:
        for idx, gt_rec in enumerate(gts, 1):
            scene = gt_rec.get("scene_id")
            pred_path = OAS_ROOT / scene / PRED_NAME
            if not pred_path.is_file():
                skipped += 1
                progress(idx, total, processed, skipped)
                continue

            if scene not in pred_cache:
                pred_cache[scene] = json.loads(pred_path.read_text())
                pred_map_cache[scene] = {
                    int(p["pred_id"]): p for p in pred_cache[scene] if "pred_id" in p
                }
                graph_engine_cache[scene] = SceneGraphQueryEngine(
                    SpatialSceneGraph.from_prediction_records(pred_cache[scene])
                )

            preds = pred_cache[scene]
            pred_map = pred_map_cache[scene]
            graph_engine = graph_engine_cache[scene]

            prompt = build_prompt(gt_rec, preds)
            response = client.chat.completions.create(
                model="qwen3-plus",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=512,
                temperature=0.0,
            )
            answer = ""
            if response.choices and response.choices[0].message:
                answer = response.choices[0].message.content or ""

            parsed = parse_response_json(answer)
            best_pred_id, graph_result, graph_error = _extract_graph_prediction(parsed, graph_engine, args.top_k)
            pred_id = _normalize_pred_id(best_pred_id)
            dist = _compute_distance_for_prediction(gt_rec, pred_map, pred_id)

            graph_top_results = graph_result.get("results") if isinstance(graph_result, dict) else None
            graph_best = graph_top_results[0] if graph_top_results else None
            out_rec = {
                "scene_id": scene,
                "ann_id": gt_rec.get("ann_id"),
                "ref_id": gt_rec.get("ref_id"),
                "object_id": gt_rec.get("object_id"),
                "label": gt_rec.get("label"),
                "description": gt_rec.get("description"),
                "best_pred_id": best_pred_id,
                "best_confidence": graph_best.get("total_score") if isinstance(graph_best, dict) else None,
                "best_direction": graph_best.get("best_direction") if isinstance(graph_best, dict) else None,
                "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
                "parsed_chain": parsed.get("parsed_chain") if isinstance(parsed, dict) else None,
                "parser_confidence": parsed.get("parser_confidence") if isinstance(parsed, dict) else None,
                "skip_due_to_size": parsed.get("skip_due_to_size") if isinstance(parsed, dict) else None,
                "graph_query": graph_result.get("query") if isinstance(graph_result, dict) else None,
                "graph_top_results": graph_top_results,
                "graph_llm_context": graph_result.get("llm_context") if isinstance(graph_result, dict) else None,
                "graph_error": graph_error,
                "raw_response": answer.strip(),
                "distance": dist,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if isinstance(graph_best, dict):
                print(
                    json.dumps(
                        {
                            "scene_id": scene,
                            "best_pred_id": graph_best.get("pred_id"),
                            "best_score": graph_best.get("total_score"),
                            "best_direction": graph_best.get("best_direction"),
                            "parsed_chain": parsed.get("parsed_chain") if isinstance(parsed, dict) else None,
                        },
                        ensure_ascii=False,
                    )
                )
            elif parsed is not None:
                print(json.dumps(parsed, ensure_ascii=False))
            else:
                print(answer.strip())

            dists.append(dist)
            processed += 1
            running_metrics = _summarize_dists(dists)
            if running_metrics is not None:
                mean_dist_run, acc30_run, acc50_run, valid_run, total_run = running_metrics
                mean_part = f"{mean_dist_run:.4f}" if mean_dist_run is not None else "n/a"
                acc30_part = f"{acc30_run:.4f}" if acc30_run is not None else "n/a"
                acc50_part = f"{acc50_run:.4f}" if acc50_run is not None else "n/a"
                print(
                    f"Running meanDist={mean_part}, Acc@0.3={acc30_part}, Acc@0.5={acc50_part} "
                    f"(valid={valid_run}/{total_run})"
                )
            progress(idx, total, processed, skipped)

            if args.print_every and processed % args.print_every == 0:
                print()
                print(f"[{processed}/{total}] processed (skipped={skipped})")
                progress(idx, total, processed, skipped)

    print()
    print(f"Done. processed={processed}, skipped={skipped}, output={args.out}")
    metrics = _summarize_dists(dists)
    if metrics is None:
        print("Eval: no processed samples to evaluate.")
    else:
        mean_dist, acc30, acc50, valid_total, total = metrics
        mean_part = f"{mean_dist:.4f}" if mean_dist is not None else "n/a"
        acc30_part = f"{acc30:.4f}" if acc30 is not None else "n/a"
        acc50_part = f"{acc50:.4f}" if acc50 is not None else "n/a"
        print(
            f"Eval: meanDist={mean_part}, Acc@0.3={acc30_part}, Acc@0.5={acc50_part} "
            f"(valid={valid_total}/{total})"
        )
        metrics_rec = {
            "processed": processed,
            "skipped": skipped,
            "mean_dist": mean_dist,
            "acc30": acc30,
            "acc50": acc50,
            "valid_count": valid_total,
            "total_count": total,
            "input_gt": len(gts),
            "output_jsonl": str(args.out),
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics_rec, ensure_ascii=False, indent=2))
        print(f"Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
