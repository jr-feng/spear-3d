import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="https://api.silra.cn/v1",
    api_key="sk-xLHvbrzWN7r8IxESmMqnzYlAr2H5LavoqfjbTDQJyZn0J4pv",  # ModelScope Token
)

# 可调参数
OAS_ROOT = Path("output_test/nr3d")
GT_PATH = Path("./nr3d/nr3d_gt_bboxes_matched.json")
PRED_NAME = "pred_bboxes.json"
MAX_GT = 500
PROGRESS_BAR_WIDTH = 30

DIST_THRESH_30 = 0.3
DIST_THRESH_50 = 0.5


def build_prompt(gt_rec, preds):
    """
    Prompt 仅让 LLM 负责“语义解析”（parsed_chain + best_direction），
    最终 best_pred_id 由本脚本中的几何求解器确定。
    """
    scene = gt_rec["scene_id"]
    desc = gt_rec.get("description") or gt_rec.get("gt_description") or ""

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
                      "best_pred_id":null,
                      "best_confidence":0.0,
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
                      "best_pred_id":null,
                      "best_confidence":0.0,
                      "reason":"middle among three upper cabinets right of refrigerator",
                      "parsed_chain":[
                        {"role":"ref","class":"refrigerator","constraints":[]},
                        {"role":"target","class":"cabinet","constraints":["upper","right of ref","middle of group of three"]}
                      ]
                    }
                    """

    prompt = f"""You are a 3D referring expression parser.
                    The input is a list of candidate instances (pred_id / class_name / center / size / color_rgb)
                    and a natural language target description.

                    Your PRIMARY task is to extract a clean, structured reasoning chain (parsed_chain) and a viewing
                    direction (best_direction). A separate geometric solver will choose the final best_pred_id.

                    Allowed numeric features (you MAY reference them qualitatively in reason but MUST NOT recompute them):
                    - class_name (category)
                    - center (x, y, z)
                    - size (x, y, z)
                    - color_rgb (coarse color only)

                    Coordinate convention (world frame):
                    - x: left(-x) / right(+x)
                    - y: back(-y) / front(+y)
                    - z: down(-z) / up(+z)

                    Viewer-centric frame:
                    - You MUST choose best_direction in {{"+x","-x","+y","-y"}}.
                    - This is the direction the human is facing.
                    - Interpret left/right/front/back using this viewer frame only, not raw axes.

                    Rules:
                    1) Extract target class + reference classes from the description.
                    2) Map description classes to candidate class_name using synonyms, plural, common paraphrases and close variants.
                       Prefer over-matching to under-matching (e.g., 'cabinet' vs 'kitchen cabinet', 'sofa' vs 'couch').
                    3) parsed_chain MUST contain:
                       - one 'target' entry: {{
                           "role": "target",
                           "class": "<target class>",
                           "constraints": ["..."]
                         }}
                       - zero or more 'ref' entries: {{
                           "role": "ref",
                           "class": "<reference class>",
                           "constraints": ["..."]
                         }}
                    4) Constraints should be short phrases capturing RELATIONS and properties, e.g.:
                       - "left of ref", "right of ref", "in front of ref", "behind ref"
                       - "next to ref", "near ref", "closest to ref"
                       - "on top of ref", "under ref"
                       - "upper", "lower", "bottom of stack", "middle of stack"
                       - "in corner", "frontmost", "backmost", "leftmost", "rightmost"
                    5) Do NOT try to pick the best_pred_id. Set best_pred_id to null and best_confidence to 0.0.
                       The geometric solver will choose best_pred_id.

                    Keep the JSON compact:
                    - reason: at most 1 short sentence summarizing the main constraints.
                    - parsed_chain: minimal but sufficient constraints; avoid redundant text.

                    Output STRICTLY valid JSON only (no extra text):
                    {{
                      "scene_id": "{scene}",
                      "best_direction": "+x/-x/+y/-y or null",
                      "best_pred_id": null,
                      "best_confidence": 0.0,
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


def _center_from_box(box):
    if box is None:
        return None
    b_min, b_max = box
    return [(b_min[i] + b_max[i]) / 2.0 for i in range(3)]


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


# ---------- 几何求解相关 ----------

def world_to_view(center, best_direction):
    """
    将世界坐标 (x,y,z) 转为观察者坐标 (u,v,w)，
    方便统一处理 left/right/front/back。

    约定：在 viewer 坐标系中
    - front  对应 +u
    - right  对应 +v
    - left   对应 -v
    """
    x, y, z = center
    if best_direction == "+x":
        # 面向 +x: front=+x, right=-y
        u, v, w = x, -y, z
    elif best_direction == "-x":
        # 面向 -x: front=-x, right=+y
        u, v, w = -x, y, z
    elif best_direction == "+y":
        # 面向 +y: front=+y, right=+x
        u, v, w = y, x, z
    elif best_direction == "-y":
        # 面向 -y: front=-y, right=-x
        u, v, w = -y, -x, z
    else:
        # 未指定方向时退回世界坐标
        u, v, w = x, y, z
    return (u, v, w)


def xy_dist(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def euclid3(c1, c2):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dz = c1[2] - c2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _norm_class(s):
    return (s or "").strip().lower()


def _build_preds_by_class(preds):
    by_cls = defaultdict(list)
    for p in preds:
        cls = _norm_class(p.get("class_name"))
        if cls:
            by_cls[cls].append(p)
    return by_cls


def _resolve_class_name(desc_class, available_classes):
    """
    简单的类名归一：先 exact，再子串匹配。
    """
    if not desc_class:
        return None
    t = _norm_class(desc_class)
    if t in available_classes:
        return t
    # cabinet vs kitchen cabinet 等
    subs = [c for c in available_classes if t in c or c in t]
    if len(subs) == 1:
        return subs[0]
    return None


def score_candidate(target_rec, cand_group, preds_by_class, parsed_chain, best_direction):
    """
    对单个候选 target 计算一个几何分数。
    目前只实现一些常见关系：closest/near/next to, left/right/front/back, above/below/on top of。
    """
    t_center = _pred_center_from_record(target_rec)
    if t_center is None:
        return -1e9
    t_view = world_to_view(t_center, best_direction)

    # 解析 chain
    target_elem = None
    ref_elems = []
    for elem in parsed_chain or []:
        role = (elem.get("role") or "").lower()
        if role == "target" and target_elem is None:
            target_elem = elem
        elif role == "ref":
            ref_elems.append(elem)

    constraints = (target_elem or {}).get("constraints") or []

    # 构造 ref 的 candidate centers
    ref_candidates = []  # list of (ref_elem, ref_center)
    for ref in ref_elems:
        r_cls_raw = ref.get("class")
        cls_key = _resolve_class_name(r_cls_raw, preds_by_class.keys())
        if not cls_key:
            continue
        for p in preds_by_class[cls_key]:
            c = _pred_center_from_record(p)
            if c is not None:
                ref_candidates.append((ref, c))

    score = 0.0

    # 1) 自身约束（只看 target group，简单支持 frontmost/backmost/leftmost/rightmost/upper/lower）
    for c in constraints:
        c_low = c.lower()

        # front/back/left/right most 相对于同类 group（viewer 坐标）
        if any(k in c_low for k in ["frontmost", "backmost", "leftmost", "rightmost"]):
            # 计算 group 的 u,v
            cand_view = [world_to_view(_pred_center_from_record(p), best_direction) for p in cand_group]
            cand_view = [cv for cv in cand_view if cv is not None]
            if cand_view:
                tu, tv, _ = t_view
                us = [cv[0] for cv in cand_view]
                vs = [cv[1] for cv in cand_view]
                if "frontmost" in c_low:
                    # u 越大分越高
                    score += (tu - min(us)) / (max(us) - min(us) + 1e-3)
                if "backmost" in c_low:
                    score += (max(us) - tu) / (max(us) - min(us) + 1e-3)
                if "rightmost" in c_low:
                    score += (tv - min(vs)) / (max(vs) - min(vs) + 1e-3)
                if "leftmost" in c_low:
                    score += (max(vs) - tv) / (max(vs) - min(vs) + 1e-3)

        # 简单 upper/lower: z 大/小
        if "upper" in c_low or "high up" in c_low:
            zs = [_pred_center_from_record(p)[2] for p in cand_group if _pred_center_from_record(p) is not None]
            if zs:
                z = t_center[2]
                score += (z - min(zs)) / (max(zs) - min(zs) + 1e-3)
        if "lower" in c_low or "bottom" in c_low:
            zs = [_pred_center_from_record(p)[2] for p in cand_group if _pred_center_from_record(p) is not None]
            if zs:
                z = t_center[2]
                score += (max(zs) - z) / (max(zs) - min(zs) + 1e-3)

    # 2) 依赖 ref 的约束
    # 如果没有 ref，就不加这部分分数
    for c in constraints:
        c_low = c.lower()
        if not ref_candidates:
            continue

        # 聚合：对所有 ref 求最优
        best_local = 0.0
        for _, r_center in ref_candidates:
            r_view = world_to_view(r_center, best_direction)
            local = 0.0
            d_xy = xy_dist(t_center, r_center)
            du = t_view[0] - r_view[0]
            dv = t_view[1] - r_view[1]
            dz = t_center[2] - r_center[2]

            # 近/远
            if "closest" in c_low or "nearest" in c_low:
                local += -d_xy
            if "near" in c_low or "next to" in c_low or "beside" in c_low:
                if d_xy < 0.5:
                    local += 1.0
                elif d_xy < 1.0:
                    local += 0.2
                else:
                    local -= 0.5

            # left/right
            if "left" in c_low and "right" not in c_low:
                local += -dv
            if "right" in c_low and "left" not in c_low:
                local += dv

            # front/back
            if "in front" in c_low or ("front" in c_low and "back" not in c_low):
                local += du
            if "behind" in c_low or ("back" in c_low and "front" not in c_low):
                local += -du

            # above / below / on top / under
            if "on top" in c_low or "above" in c_low:
                local += dz - d_xy * 0.2
            if "under" in c_low or "below" in c_low:
                local += -dz - d_xy * 0.2

            # corner: 距离场景角落小
            if "corner" in c_low:
                # 粗略: 用 ref 本身当 corner anchor，距离越小越好
                local += -d_xy

            if local > best_local:
                best_local = local

        score += best_local

    return score


def choose_best_pred_id(preds, parsed_chain, best_direction):
    """
    几何求解入口：
    - 先用 parsed_chain 里的 target class 选候选集合；
    - 再对候选打分，选 score 最高的 pred_id。
    """
    if not preds:
        return None

    preds_by_class = _build_preds_by_class(preds)

    # 取第一个 target entry
    target_elem = None
    for elem in parsed_chain or []:
        if (elem.get("role") or "").lower() == "target":
            target_elem = elem
            break

    target_class = (target_elem or {}).get("class")
    target_cls_key = _resolve_class_name(target_class, preds_by_class.keys()) if target_class else None

    if target_cls_key and preds_by_class.get(target_cls_key):
        cand_group = preds_by_class[target_cls_key]
    else:
        # 找不到匹配类时，退回全体候选
        cand_group = preds

    scored = []
    for p in cand_group:
        s = score_candidate(p, cand_group, preds_by_class, parsed_chain, best_direction)
        scored.append((s, int(p["pred_id"])))

    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


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


def main():
    ap = argparse.ArgumentParser(description="LLM parse + geometric solver for NR3D.")
    ap.add_argument("--scene", default=None, help="Only process a single scene_id")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max number of GT descriptions to process (default: 500, cap at 500)",
    )
    ap.add_argument("--out", type=Path, default=Path("output_test/nr3d/llm_geo_outputs.json"), help="Output JSONL path")
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default="output_test/nr3d/llm_geo_metrics.json",
        help="Output metrics JSON path",
    )
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
    dists = []
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
                model="qwen3-plus",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                extra_body=extra_body,
                max_tokens=512,
            )

            answer = ""
            if response.choices and response.choices[0].message:
                answer = response.choices[0].message.content or ""
            parsed = parse_response_json(answer)

            llm_best_pred_id = None
            best_direction = None
            parsed_chain = None
            if isinstance(parsed, dict):
                llm_best_pred_id = parsed.get("best_pred_id", None)
                best_direction = parsed.get("best_direction")
                parsed_chain = parsed.get("parsed_chain") or []

            # 用几何求解器选最终 pred_id
            geo_pred_id = choose_best_pred_id(preds, parsed_chain, best_direction)
            dist = _compute_distance_for_prediction(gt_rec, pred_map, geo_pred_id)

            out_rec = {
                "scene_id": scene,
                "ann_id": gt_rec.get("ann_id"),
                "ref_id": gt_rec.get("ref_id"),
                "object_id": gt_rec.get("object_id"),
                "label": gt_rec.get("label"),
                "description": gt_rec.get("description"),
                # 几何求解器最终结果
                "best_pred_id": geo_pred_id,
                "distance": dist,
                # LLM 原始解析信息
                "llm_best_pred_id": llm_best_pred_id,
                "best_confidence": parsed.get("best_confidence") if isinstance(parsed, dict) else None,
                "best_direction": best_direction,
                "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
                "parsed_chain": parsed_chain,
                "raw_response": answer.strip(),
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


if __name__ == "__main__":
    main()
