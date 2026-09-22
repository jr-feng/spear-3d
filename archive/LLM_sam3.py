import argparse
import json
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url='https://api.silra.cn/v1',
    api_key='sk-U8XPehUXqrsoFfzZbULC7tRWrVBOTLOB9KS1USjdTvnj2stT',  # ModelScope Token
)

# 可调参数
OAS_ROOT = Path("output_test/nr3d")
GT_PATH = Path("./nr3d/nr3d_gt_bboxes_matched.json")
#GT_PATH = Path("./nr3d/nr3d_pred_bboxes_matched.json")
PRED_NAME = "pred_bboxes.json"
MAX_GT = 500
PROGRESS_BAR_WIDTH = 30


def build_prompt(gt_rec, preds):
    scene = gt_rec["scene_id"]
    desc = gt_rec.get("description", "")
    #desc = gt_rec.get("gt_description", "")
    # 只保留指定字段
    minimal = [
        {
            "pred_id": int(p["pred_id"]),
            "class_name": p.get("class_name"),
            "center": p.get("center"),
            "size": p.get("size"),
            "color_rgb": p.get("color_rgb"),
        }
        for p in preds
    ]

    example_section = """
                    Example:
                    Description: "the chair to the right of the trash can."
                    Example output:
                    {
                      "scene_id":"scene0000_00",
                      "best_direction":"+y",
                      "best_pred_id":12,
                      "best_confidence":0.78,
                      "reason":"right side and nearby",
                      "parsed_chain":[
                        {"role":"ref","class":"trash can","constraints":["right-of target"]},
                        {"role":"target","class":"chair","constraints":["right of ref"]}
                      ]
                    }
                    """

    prompt = f"""
                You are a 3D instance grounding assistant. The input is a list of candidate instances
                (pred_id / class_name / center / size / color_rgb) and a target description.
                Only use the "available features" and ignore material/texture/brand/pattern/text/details.

                Available features:
                - class_name (category)
                - center (center point)
                - size (bounding box size, x/y/z)
                - color_rgb (mean color, only for coarse color cues)

                Unavailable features (ignore them; do NOT lower confidence because of them):
                - material/texture/brand/pattern/text/details (wooden/leather/metal/glass, etc.)

                Coordinate convention (relative relations only):
                - x axis: left(-x) / right(+x)
                - y axis: back(-y) / front(+y)
                - z axis: down(-z) / up(+z)

                Tasks:
                1) Parse the description, extract the target class and any reference objects, then output parsed_chain.
                2) Allow synonym matching when mapping description classes to candidates
                   (trash can≈ashcan≈garbage bin≈waste bin, cabinet≈cupboard, sofa≈couch, tv≈television, etc.).
                   Treat plural/singular and minor spelling variants as equivalent.
                3) Only choose among candidates of the same class; if none exist, output null.
                4) Relation rules (use only center/size):
                   - left/right/front/back/behind: compare center x/y.
                   - leftmost/rightmost/frontmost/backmost: pick x/y extrema among target class.
                   - next to/near/beside/adjacent/by/around: minimum center distance (prefer xy distance).
                   - closest/nearest/farthest/furthest: min/max distance to the reference.
                   - between A and B: target center near the midpoint of A/B and with similar distances to A/B.
                   - middle/center of room: target center closest to the mean of all candidate centers.
                   - end of row/line: within an approximately collinear group, choose the endpoint along the main axis.
                   - in front of / behind: compare y direction.
                   - on top of / above / over: target has higher z; "on top of" also requires closer xy distance.
                   - under / below / beneath: target has lower z; if "under X", it should be near X.
                   - against wall / on the wall / corner / side of room: near global boundary (x/y extrema);
                     "side of room with X" means on the same side as X (same sign of x or y vs the mean center).
                   - with/has/with X on it: target is closer to X; if on/under appears, add vertical relation.
                   - group of two/three: first find the closest same-class group, then apply left/right/front/back within it.
                   - size adjectives: tall/short -> compare size_z; big/small/large -> compare volume;
                     long/wide/thin -> compare max/min size axis.
                5) Color (optional): only use when the description explicitly mentions a color; color_rgb is coarse
                   (white/black/brown/gray/blue/red/green/yellow/orange/pink/purple). dark/light is only a brightness bias.
                6) If constraints conflict or cannot be determined, best_pred_id = null, best_confidence = 0.
                7) Output constraints: output JSON only, no extra text; reason must be a short phrase (<=20 words).

                Output JSON (only this structure):
                {{
                  "scene_id": "{scene}",
                  "best_direction": "+x/-x/+y/-y or null",
                  "best_pred_id": <int or null>,
                  "best_confidence": 0-1,
                  "reason": "...",
                  "parsed_chain": [
                    {{"role": "ref/target", "class": "...", "constraints": ["..."]}},
                    ...
                  ]
                }}

                Target description:
                - text: "{desc}"

                Candidate instances (minimal fields):
                {json.dumps(minimal, ensure_ascii=False, indent=2)}

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


def _iou_boxes(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    b1_min, b1_max = box1
    b2_min, b2_max = box2
    inter_min = [max(b1_min[i], b2_min[i]) for i in range(3)]
    inter_max = [min(b1_max[i], b2_max[i]) for i in range(3)]
    inter_size = [max(0.0, inter_max[i] - inter_min[i]) for i in range(3)]
    inter_vol = inter_size[0] * inter_size[1] * inter_size[2]
    vol1 = (
        max(0.0, b1_max[0] - b1_min[0])
        * max(0.0, b1_max[1] - b1_min[1])
        * max(0.0, b1_max[2] - b1_min[2])
    )
    vol2 = (
        max(0.0, b2_max[0] - b2_min[0])
        * max(0.0, b2_max[1] - b2_min[1])
        * max(0.0, b2_max[2] - b2_min[2])
    )
    union = vol1 + vol2 - inter_vol
    if union <= 1e-8:
        return 0.0
    return inter_vol / union


def _normalize_pred_id(pred_id):
    if pred_id is None:
        return None
    try:
        return int(pred_id)
    except (TypeError, ValueError):
        return None


def _pred_box_from_record(pred_rec):
    if not pred_rec:
        return None
    box = _bbox_from_min_max(pred_rec.get("box_min"), pred_rec.get("box_max"))
    if box is not None:
        return box
    return _bbox_from_center_size(pred_rec.get("center"), pred_rec.get("size"))


def _compute_iou_for_prediction(gt_rec, pred_map, pred_id):
    gt_box = _bbox_from_center_size(gt_rec.get("center"), gt_rec.get("size"))
    pred_rec = pred_map.get(pred_id) if pred_id is not None else None
    pred_box = _pred_box_from_record(pred_rec)
    return _iou_boxes(gt_box, pred_box)


def _summarize_ious(ious):
    if not ious:
        return None
    total = len(ious)
    miou = sum(ious) / total
    acc25 = sum(1 for i in ious if i >= 0.25) / total
    acc50 = sum(1 for i in ious if i >= 0.5) / total
    return miou, acc25, acc50


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


def main():
    ap = argparse.ArgumentParser(description="Run LLM for each GT description and write results to JSONL.")
    ap.add_argument("--scene", default=None, help="Only process a single scene_id")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max number of GT descriptions to process (default: 500, cap at 500)",
    )
    ap.add_argument("--out", type=Path, default=Path("output_test/nr3d/llm_outputs.json"), help="Output JSONL path")
    ap.add_argument("--metrics-out", type=Path, default="output_test/nr3d/llm_metrics.json", help="Output metrics JSON path")
    ap.add_argument("--print-every", type=int, default=50, help="Progress print interval (0 to disable)")
    args = ap.parse_args()
    if args.metrics_out is None:
        args.metrics_out = args.out.with_suffix(".metrics.json")

    gts = json.loads(GT_PATH.read_text())
    if args.scene:
        gts = [rec for rec in gts if rec.get("scene_id") == args.scene]
    max_limit = MAX_GT if args.max is None else min(args.max, MAX_GT)
    if max_limit is not None:
        gts = gts[:max_limit]

    if not gts:
        raise SystemExit("没有匹配到可处理的 GT 描述记录")

    pred_cache = {}
    pred_map_cache = {}
    processed = 0
    skipped = 0
    total = len(gts)
    ious = []
    progress = _make_progress_printer()

    extra_body = {"enable_thinking": False}
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
            preds = pred_cache[scene]
            pred_map = pred_map_cache[scene]

            prompt = build_prompt(gt_rec, preds)
            response = client.chat.completions.create(
                model= "qwen3-plus",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                extra_body=extra_body,
                max_tokens=512,
            )
            answer = ""
            if response.choices and response.choices[0].message:
                answer = response.choices[0].message.content or ""
            parsed = parse_response_json(answer)
            best_pred_id = None
            if isinstance(parsed, dict):
                best_pred_id = parsed.get("best_pred_id", None)
            pred_id = _normalize_pred_id(best_pred_id)
            iou = _compute_iou_for_prediction(gt_rec, pred_map, pred_id)
            out_rec = {
                "scene_id": scene,
                "ann_id": gt_rec.get("ann_id"),
                "ref_id": gt_rec.get("ref_id"),
                "object_id": gt_rec.get("object_id"),
                "label": gt_rec.get("label"),
                "description": gt_rec.get("description"),
                "best_pred_id": best_pred_id,
                "best_confidence": parsed.get("best_confidence") if isinstance(parsed, dict) else None,
                "best_direction": parsed.get("best_direction") if isinstance(parsed, dict) else None,
                "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
                "parsed_chain": parsed.get("parsed_chain") if isinstance(parsed, dict) else None,
                "raw_response": answer.strip(),
                "iou": iou,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            if parsed is not None:
                print(json.dumps(parsed, ensure_ascii=False))
            else:
                print(answer.strip())
            ious.append(iou)
            processed += 1
            running_metrics = _summarize_ious(ious)
            if running_metrics is not None:
                _, acc25_run, acc50_run = running_metrics
                print(f"Running Acc@0.25={acc25_run:.4f}, Acc@0.5={acc50_run:.4f}")
            progress(idx, total, processed, skipped)

            if args.print_every and processed % args.print_every == 0:
                print()
                print(f"[{processed}/{total}] processed (skipped={skipped})")
                progress(idx, total, processed, skipped)

    print()
    print(f"Done. processed={processed}, skipped={skipped}, output={args.out}")
    metrics = _summarize_ious(ious)
    if metrics is None:
        print("Eval: no processed samples to evaluate.")
    else:
        miou, acc25, acc50 = metrics
        print(f"Eval: mIoU={miou:.4f}, Acc@0.25={acc25:.4f}, Acc@0.5={acc50:.4f}")
        metrics_rec = {
            "processed": processed,
            "skipped": skipped,
            "miou": miou,
            "acc25": acc25,
            "acc50": acc50,
            "input_gt": len(gts),
            "output_jsonl": str(args.out),
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics_rec, ensure_ascii=False, indent=2))
        print(f"Metrics saved to {args.metrics_out}")


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


if __name__ == "__main__":
    main()
