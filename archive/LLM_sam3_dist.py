import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on runtime environment
    OpenAI = None

# 可调参数
OAS_ROOT = Path("output_test/nr3d")
#GT_PATH = Path("./sr3d/sr3d_gt_bboxes_matched.json")
GT_PATH = Path("./nr3d/nr3d_pred_bboxes_matched.json")
PRED_NAME = "pred_bboxes.json"
MAX_GT = 500
PROGRESS_BAR_WIDTH = 30

#########qwen3_plus
# DEFAULT_OPENAI_BASE_URL = "https://api.silra.cn/v1"
# DEFAULT_OPENAI_API_KEY = "sk-CvP9I6FP8Kzv06XzCYdu1WnLQmCNbVuTosIWZqVdJcAEI2Ib"
# DEFAULT_LLM_MODEL = "qwen3-plus"

########deepseekv3.2
# DEFAULT_OPENAI_BASE_URL = "https://api.silra.cn/v1"
# DEFAULT_OPENAI_API_KEY = "sk-S2ZncuMuHCXO3sIGla2efB4iYYBWXUJ6MHxhqui67NbcMzRD"
# DEFAULT_LLM_MODEL = "deepseek-v3.2"

###########glm-5.1
DEFAULT_OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_OPENAI_API_KEY = "25580d8d47a1463b89d3975f89fbc908.pgkPCgAd7YbhJEvh"
DEFAULT_LLM_MODEL = "GLM-5"




def build_prompt(gt_rec, preds):
    scene = gt_rec["scene_id"]
    desc = gt_rec.get("description") or gt_rec.get("gt_description") or ""
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
                      "best_direction":"+x",
                      "best_pred_id":12,
                      "best_confidence":0.78,
                      "reason":"right of trash can",
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
                      "best_pred_id":24,
                      "best_confidence":0.72,
                      "reason":"middle among three upper cabinets right of refrigerator",
                      "parsed_chain":[
                        {"role":"ref","class":"refrigerator","constraints":[]},
                        {"role":"target","class":"cabinet","constraints":["upper","right of ref","middle of group of three"]}
                      ]
                    }
                    """
    prompt = f"""You are a 3D instance grounding assistant.
        The input is a list of candidate instances (pred_id / class_name / center / size / color_rgb) and a target description.
        You must pick best_pred_id ONLY from the candidate list.

        Available features:
        - class_name (category)
        - center (center point)
        - size (bounding box size, x/y/z)
        - color_rgb (mean color, only for coarse color cues)

        Unavailable features (ignore them; do NOT lower confidence because of them):
        - material/texture/brand/pattern/text/details (wooden/leather/metal/glass, etc.)

        There is a fixed world coordinate system (x, y, z) for the scene.
        However, natural-language words like "front/back/left/right" are VIEWER-CENTRIC and depend on which way the human is facing.

        You MUST:
        - Choose a single viewing direction best_direction ∈ {{"+x","-x","+y","-y"}} that represents where the human is facing in world coordinates.
        - Interpret all "front/back/left/right" words in a viewer-centric coordinate frame defined by best_direction.

        Viewer-centric frame (in the horizontal xy-plane):
        - If best_direction = "+x": 
            front = +x, back = -x, left = +y, right = -y
        - If best_direction = "-x": 
            front = -x, back = +x, left = -y, right = +y
        - If best_direction = "+y": 
            front = +y, back = -y, left = -x, right = +x
        - If best_direction = "-y": 
            front = -y, back = +y, left = +x, right = -x

        Use this viewer-centric frame when checking:
        - "in front of", "behind", "left of", "right of", "front/back/left/right side", etc.
        Other geometric rules (near/next to/above/below, distances) still use the world coordinates (x,y,z).


        Rules:
        1) Extract target class + reference classes from the description.
        2) Map description classes to candidate class_name_norm using synonyms/plural only.
           If target class or reference class has no candidate, you MAY still choose a best_pred_id from all candidates
           with LOWER confidence, except when the description is primarily about size (see rule 11).
        3) Use candidates of the target class whenever possible. Only back off to all candidates when no target-class
           candidates exist at all.
        4) If a referenced class has multiple instances, evaluate all and pick the target that best satisfies the constraints across refs (do NOT discard).
        5) If a referenced class is missing, drop only that constraint and continue.
        6) If constraints are insufficient or conflicting, still choose the best candidate with LOWER confidence; only
           return null when the candidate list is empty OR when rule 11 applies.
        7) The `reason` MUST be concise: at most 2 sentences, no step-by-step recomputation, no long equations.
        8) Keep the entire JSON under 200 tokens. Do NOT explain each numeric calculation in detail.
       
        10) Size adjectives (BIG influence on semantics but evaluation will SKIP these cases):
                   - tall/short, taller/shorter
                   - big/small/large/tiny/huge
                   - long/wide/thin/narrow
                   - "bigger of two", "smaller of two", "largest", "smallest", "tallest", "shortest"
            If the description for the TARGET object clearly depends on such size/scale words as the MAIN distinguishing
            factor, you MUST NOT guess. In that case:
                   - Set best_pred_id = null.
                   - Set best_confidence = 0.0.
                   - In reason, briefly say that this is a size-based description and is skipped by instruction.
            (These samples will be excluded from accuracy evaluation.)
        11) Color (weak signal, lower priority than position/geometry):
                   - Only use when the description explicitly mentions a color.
                   - color_rgb is coarse (white/black/brown/gray/blue/red/green/yellow/orange/pink/purple).
                   - dark/light is only a brightness bias.
                   - When color conflicts with strong geometric relations, ALWAYS prefer the geometrically correct
                     candidate and treat color as a soft tie-breaker only when geometry is similar.

        Output JSON only (no extra text):

        {{ 
        "scene_id": "{scene}", 
        "best_direction": "+x/-x/+y/-y or null",
        "best_pred_id": <int or null>, 
        "best_confidence": 0-1, 
        "reason": "...", 
        "parsed_chain": [{{"role": "ref/target", "class": "...", "constraints": ["..."]}}] 
        }}

        Target description: "{desc}"
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


DIST_THRESH_30 = 0.3
DIST_THRESH_50 = 0.5


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


def make_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("The `openai` package is required to run this script.")
    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY),
    )


def _llm_extra_body(model: str) -> Optional[Dict[str, Any]]:
    model_l = (model or "").lower()
    if "glm-5" in model_l or "glm-4.7" in model_l:
        return {"thinking": {"type": "disabled"}}
    return None


def request_llm_json_text(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
) -> str:
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    extra_body = _llm_extra_body(model)
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message is None:
        return ""
    if choice.finish_reason == "length":
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        content = choice.message.content or ""
        return content or reasoning
    return choice.message.content or ""


def main():
    ap = argparse.ArgumentParser(description="Run LLM for each GT description and write results to JSONL.")
    ap.add_argument("--scene", default=None, help="Only process a single scene_id")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max number of GT descriptions to process (default: 500, cap at 500)",
    )
    ap.add_argument("--out", type=Path, default=Path("output_test/sr3d/llm_sr3d_no.json"), help="Output JSONL path")
    #ap.add_argument("--out", type=Path, default=Path("output_test/nr3d/deepseek_nr3d_no.json"), help="Output JSONL path")
    ap.add_argument("--metrics-out", type=Path, default=Path("output_test/sr3d/deepseek_sr3d_no.json"), help="Output metrics JSON path")
    ap.add_argument("--print-every", type=int, default=50, help="Progress print interval (0 to disable)")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--openai-base-url", default=None)
    ap.add_argument("--openai-api-key", default=None)
    args = ap.parse_args()
    if args.metrics_out is None:
        args.metrics_out = args.out.with_suffix(".metrics.json")

    gts = json.loads(GT_PATH.read_text())
    if args.scene:
        gts = [rec for rec in gts if rec.get("scene_id") == args.scene]

    # 随机抽样评估：从全部 GT 中随机打乱后取前 max_limit 条
    random.shuffle(gts)
    max_limit = MAX_GT if args.max is None else min(args.max, MAX_GT)
    if max_limit is not None:
        gts = gts[:max_limit]

    if not gts:
        raise SystemExit("没有匹配到可处理的 GT 描述记录")

    client = make_client(args.openai_base_url, args.openai_api_key)

    pred_cache = {}
    pred_map_cache = {}
    processed = 0
    skipped = 0
    total = len(gts)
    dists = []
    progress = _make_progress_printer()

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
            answer = request_llm_json_text(client, args.llm_model, prompt, max_tokens=512)
            parsed = parse_response_json(answer)
            best_pred_id = None
            if isinstance(parsed, dict):
                best_pred_id = parsed.get("best_pred_id", None)
            pred_id = _normalize_pred_id(best_pred_id)
            dist = _compute_distance_for_prediction(gt_rec, pred_map, pred_id)
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
                "distance": dist,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            if parsed is not None:
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
