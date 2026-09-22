#!/usr/bin/env python3
"""
L0 baseline: black-box LLM direct grounding (no NS-Grounding).

For each referring-expression query, the reconstructed object list
(category, ID, 3D center) and the description are sent directly to the LLM,
which selects the object ID. Evaluation reuses the SAME protocol as
LLM_sam3_query_graph_clip.py (NS-Grounding): MeanDist, Acc@0.3m/0.5m,
Coverage, SelAcc@all/@covered and error split.

Config: config/grounding_eval_qwen3_n_nr3d.yaml (same schema as NS-Grounding).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from openai import OpenAI


class LLM:
    """Black-box LLM client for direct (no-NS) grounding."""

    def __init__(
        self,
        inst_text=None,
        inst_id=None,
        inst_center_coords=None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.SYSTEM_INFO = (
            "You are a helpful assistant designed to identify the object that best "
            "matches a referring expression, given the 3D spatial coordinates of objects."
        )
        self.COOR_INFO = (
            "The 3D spatial coordinate system is defined as follows: "
            "X-axis and Y-axis represent horizontal dimensions, Z-axis represents the vertical dimension."
        )
        self.RESPONSE_FORMAT = (
            "Respond with ONLY a single JSON object and NOTHING ELSE, in the format: "
            '{"object": "<category>", "id": <integer object id>, "coordinates": [x, y, z]}.'
        )
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model or "qwen3-plus"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        # keep legacy fields for compatibility (unused by the L0 pipeline)
        self.inst_text = inst_text
        self.inst_id = inst_id
        self.inst_center_coords = inst_center_coords

    def create_openai_message(self, query, objects_info):
        """Build the messages; query: {'obs_dir': ..., 'task': ...} (kept for compatibility)."""
        user_text = (
            f"{self.COOR_INFO}\n\n"
            "Object list (one per line, format: category, ID, (x, y, z)):\n"
            f"{objects_info}\n\n"
            f"{query.get('obs_dir', '(0,0,0)')}\n"
            f"Task description: {query.get('task', '')}\n\n"
            f"{self.RESPONSE_FORMAT}\n"
            "OUTPUT RULE: ONLY output a single JSON object and NOTHING ELSE. "
        )
        return [
            {"role": "system", "content": self.SYSTEM_INFO},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

    def query(self, objects_info: str, description: str) -> str:
        """Directly ask the LLM to select the object matching `description`."""
        query = {"obs_dir": "(0,0,0)", "task": description}
        messages = self.create_openai_message(query, objects_info)
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=False, temperature=0.0
        )
        return (response.choices[0].message.content or "") if response.choices else ""

    def LLM_query(self):
        """Legacy test entry (kept for compatibility)."""
        print(self.query("", "find out all chairs"))


def parse_best_pred_id(answer: str) -> Optional[int]:
    """Best-effort extraction of the selected object id from the LLM answer."""
    if not answer:
        return None
    s = answer.find("{")
    e = answer.rfind("}")
    if s != -1 and e > s:
        try:
            obj = json.loads(answer[s : e + 1])
            for key in ("id", "ID", "pred_id", "object_id", "Predicted ID"):
                v = obj.get(key)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
        except json.JSONDecodeError:
            pass
    m = re.search(r"(?i)(?:predicted\s+)?(?:id|ID)\s*[:\s#-]*\s*(\d+)", answer)
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L0 baseline: black-box LLM direct grounding (no NS-Grounding).")
    parser.add_argument("--config", type=str, default="config/grounding_eval_qwen3_n_nr3d.yaml",
                        help="YAML config (same schema as NS-Grounding eval)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--oas-root", type=str, default=None)
    parser.add_argument("--pred-name", type=str, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--corr-thresh", type=float, default=0.5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)

    # pass 1: read --config; load YAML; pass 2: CLI overrides config, config overrides defaults
    args = parser.parse_known_args()[0]

    # lazy import: reuse the NS-Grounding evaluation helpers (same protocol)
    from LLM_sam3_query_graph_clip import (  # noqa: E402
        _compute_distance_for_prediction,
        _corresponds,
        _load_grounding_config,
        _pred_center_from_record,
        _summarize_dists,
    )

    flat = _load_grounding_config(args.config)
    if flat:
        parser.set_defaults(**flat)
    args = parser.parse_args()

    if args.out is None:
        args.out = Path("ground_ral/nr3d/qwen3_l0_outputs.jsonl")
    if args.metrics_out is None:
        args.metrics_out = Path("ground_ral/nr3d/qwen3_l0_metrics.json")

    gts = json.loads(Path(args.gt_path).read_text())
    if args.scene:
        gts = [g for g in gts if g.get("scene_id") == args.scene]
    if args.max is not None:
        gts = gts[: args.max]
    if not gts:
        raise SystemExit("no GT records")

    scene_gt_objs: Dict[str, Dict[Any, Mapping[str, Any]]] = {}
    for _g in gts:
        scene_gt_objs.setdefault(_g.get("scene_id"), {})[_g.get("object_id")] = _g

    llm = LLM(base_url=args.openai_base_url, api_key=args.openai_api_key, model=args.llm_model)

    pred_map_cache: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    processed = 0
    skipped = 0
    total = len(gts)
    dists: List[Optional[float]] = []
    n_covered = 0
    n_correct = 0
    error_cnt = {"not_reconstructed": 0, "no_prediction": 0, "wrong_selection": 0, "hallucinated": 0}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f:
        for idx, gt_rec in enumerate(gts, 1):
            scene_id = gt_rec.get("scene_id")
            pred_path = Path(args.oas_root) / scene_id / args.pred_name
            if not pred_path.is_file():
                skipped += 1
                continue
            if scene_id not in pred_map_cache:
                preds = json.loads(pred_path.read_text())
                pred_map_cache[scene_id] = {int(p["pred_id"]): p for p in preds if "pred_id" in p}
            pred_map = pred_map_cache[scene_id]

            # build the object-list text for the LLM (category, ID, (x,y,z))
            lines = []
            for pid, p in pred_map.items():
                c = _pred_center_from_record(p)
                if c is None:
                    continue
                lines.append(f"{p.get('class_name')}, {pid}, ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")
            objects_info = "\n".join(lines)
            description = gt_rec.get("description") or ""

            answer = llm.query(objects_info, description)
            best_pred_id = parse_best_pred_id(answer)
            if best_pred_id is not None and best_pred_id not in pred_map:
                best_pred_id = None  # invalid id -> treated as no prediction

            dist = _compute_distance_for_prediction(gt_rec, pred_map, best_pred_id)

            corr_thresh = args.corr_thresh
            gt_objs = scene_gt_objs.get(scene_id, {})
            covered = any(_corresponds(p, gt_rec, corr_thresh) for p in pred_map.values())
            sel_rec = pred_map.get(best_pred_id) if best_pred_id is not None else None
            if sel_rec is None:
                error_type, correct = "no_prediction", False
            elif _corresponds(sel_rec, gt_rec, corr_thresh):
                error_type, correct = "correct", True
            else:
                other = next(
                    (g for oid, g in gt_objs.items()
                     if oid != gt_rec.get("object_id") and _corresponds(sel_rec, g, corr_thresh)),
                    None,
                )
                error_type, correct = ("wrong_selection" if other is not None else "hallucinated"), False

            if covered:
                n_covered += 1
            if correct:
                n_correct += 1
            if not covered:
                error_cnt["not_reconstructed"] += 1
            elif not correct:
                error_cnt[error_type] += 1

            out_rec = {
                "scene_id": scene_id,
                "ann_id": gt_rec.get("ann_id"),
                "ref_id": gt_rec.get("ref_id"),
                "object_id": gt_rec.get("object_id"),
                "label": gt_rec.get("label"),
                "description": description,
                "best_pred_id": best_pred_id,
                "distance": dist,
                "covered": covered,
                "sel_correct": correct,
                "error_type": error_type,
                "raw_answer": answer,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            dists.append(dist)
            processed += 1

            running_metrics = _summarize_dists(dists)
            if running_metrics is not None:
                mean_dist_run, acc30_run, acc50_run, valid_run, total_run = running_metrics
                mean_part = f"{mean_dist_run:.4f}" if mean_dist_run is not None else "n/a"
                acc30_part = f"{acc30_run:.4f}" if acc30_run is not None else "n/a"
                acc50_part = f"{acc50_run:.4f}" if acc50_run is not None else "n/a"
                print(
                    f"Eval: meanDist={mean_part}, Acc@0.3m={acc30_part}, Acc@0.5m={acc50_part} "
                    f"| Coverage={n_covered / max(processed, 1):.4f}, "
                    f"SelAcc@all={n_correct / max(processed, 1):.4f}, "
                    f"SelAcc@covered={n_correct / max(n_covered, 1):.4f} "
                    f"(valid={valid_run}/{total_run})"
                )

            if args.print_every and processed % args.print_every == 0:
                print(f"[{processed}/{total}] processed (skipped={skipped})")

    metrics = _summarize_dists(dists)
    if metrics is None:
        print("Eval: no processed samples.")
        return
    mean_dist, acc30, acc50, valid_total, total_count = metrics
    print(f"Eval: meanDist={mean_dist:.4f}, Acc@0.3m={acc30:.4f}, Acc@0.5m={acc50:.4f} "
          f"| Coverage={n_covered / max(processed, 1):.4f}, "
          f"SelAcc@all={n_correct / max(processed, 1):.4f}, "
          f"SelAcc@covered={n_correct / max(n_covered, 1):.4f} "
          f"(valid={valid_total}/{total_count})")
    metrics_rec = {
        "processed": processed,
        "skipped": skipped,
        "mean_dist": mean_dist,
        "acc03m": acc30,
        "acc05m": acc50,
        "coverage": n_covered / max(processed, 1),
        "selacc_all": n_correct / max(processed, 1),
        "selacc_covered": n_correct / max(n_covered, 1),
        "error_split": error_cnt,
        "valid_count": valid_total,
        "total_count": total_count,
        "input_gt": len(gts),
        "output_jsonl": str(args.out),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics_rec, ensure_ascii=False, indent=2))
    print(f"Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
