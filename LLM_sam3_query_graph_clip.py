#!/usr/bin/env python3
"""
LLM query-graph grounding with CLIP-based class matching and scene-graph search.

Pipeline:
1. LLM parses a referring expression into a structured query graph.
2. Code builds a 3D spatial scene model from predicted instances.
3. Query node classes are resolved against scene class names with CLIP text embeddings.
4. A beam-search matcher finds top-k scene assignments for the query graph.
5. An optional LLM reranks the shortlist using structured geometric evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None

try:
    import open_clip
except ImportError:  # pragma: no cover - depends on runtime environment
    open_clip = None


try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on runtime environment
    OpenAI = None

from spatial_scene_graph import SceneObject, SpatialSceneGraph

# class-agreement correspondence for the grounding protocol (SelAcc/Coverage)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
try:
    from match_nr3d_gt_to_pred import normalize_label as _normalize_class, DEFAULT_SYNONYM_GROUPS as _SYNONYM_GROUPS
except Exception:  # pragma: no cover - fallback to local normalizer
    _normalize_class = None
    _SYNONYM_GROUPS = []


OAS_ROOT = Path("output_test/nr3d")
GT_PATH = Path("./nr3d/nr3d_gt_bboxes_matched.json")
# GT_PATH = Path("./sr3d/sr3d_gt_bboxes_matched.json")
PRED_NAME = "pred_bboxes.json"
MAX_GT = 500
PROGRESS_BAR_WIDTH = 30

DIST_THRESH_30 = 0.3
DIST_THRESH_50 = 0.5
ALL_VIEWER_DIRECTIONS = ("+x", "-x", "+y", "-y")

# Global knob for graph scoring.
# GRAPH_BALANCE is the node-side share in [0, 1]; relation-side share is 1 - GRAPH_BALANCE.
# Set GRAPH_BALANCE=0.5 to match the previous equal weighting behavior.
GRAPH_BALANCE = 0

# --- spatial relation scorer library (paper Sec. III-B.4) -------------------
# Each of the 7 hand-crafted relation scorers owns a subset of normalized rel_types.
SCORER_RELTYPES: Dict[str, Tuple[str, ...]] = {
    "dir": ("left_of", "right_of", "front_of", "behind"),          # s_dir
    "near": ("near",),                                              # s_near
    "rank": ("closest_to", "farthest_from"),                        # s_rank
    "vert": ("above", "below"),                                     # s_vert
    "contact": ("on_top_of", "under"),                              # s_contact
    "side": ("same_side_as",),                                      # s_side
    "between": ("between",),                                        # s_between
}
SCORER_NAMES = tuple(SCORER_RELTYPES.keys())
_RELTYPE_TO_SCORER: Dict[str, str] = {
    rt: name for name, rels in SCORER_RELTYPES.items() for rt in rels
}

# Existing role-specific node prior weights.
TARGET_NODE_ROLE_WEIGHT = 2.0
REF_NODE_ROLE_WEIGHT = 1.0


#########qwen3_plus
DEFAULT_OPENAI_BASE_URL = "https://api.silra.cn/v1"
DEFAULT_OPENAI_API_KEY = "sk-cqGLoCze45M7GqOeV2DDyXvYB06dujekBIZAWVjc0eE5cSfb"

########deepseekv3.2
# DEFAULT_OPENAI_BASE_URL="https://api.silra.cn/v1"
# DEFAULT_OPENAI_API_KEY="sk-S2ZncuMuHCXO3sIGla2efB4iYYBWXUJ6MHxhqui67NbcMzRD" # ModelScope Token

###########glm-5.1
# DEFAULT_OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
# DEFAULT_OPENAI_API_KEY = "25580d8d47a1463b89d3975f89fbc908.pgkPCgAd7YbhJEvh"
# DEFAULT_LLM_MODEL ='GLM-5'

############
DEFAULT_LLM_MODEL = "qwen3-plus"

# DEFAULT_LLM_MODEL ='deepseek-v3.2'




DEFAULT_CLIP_MODEL = "ViT-H-14"
DEFAULT_CLIP_PRETRAINED = Path("models/open_clip_pytorch_model.bin")

EPS = 1e-8

RELATION_ALIASES = {
    "left": "left_of",
    "left_of": "left_of",
    "right": "right_of",
    "right_of": "right_of",
    "front": "front_of",
    "front_of": "front_of",
    "in_front_of": "front_of",
    "behind": "behind",
    "back_of": "behind",
    "near": "near",
    "next_to": "near",
    "beside": "near",
    "adjacent_to": "near",
    "closest": "closest_to",
    "closest_to": "closest_to",
    "nearest": "closest_to",
    "nearest_to": "closest_to",
    "farthest": "farthest_from",
    "farthest_from": "farthest_from",
    "furthest": "farthest_from",
    "furthest_from": "farthest_from",
    "above": "above",
    "below": "below",
    "under": "under",
    "on_top_of": "on_top_of",
    "same_side_as": "same_side_as",
    "side_of_room_with": "same_side_as",
    "with": "with_object",
    "with_object": "with_object",
    "has_on_it": "has_on_it",
    "between": "between",
}

UNARY_ATTR_ALIASES = {
    "upper": "upper",
    "higher": "upper",
    "high_up": "upper",
    "lower": "lower",
    "bottom": "lower",
    "leftmost": "leftmost",
    "rightmost": "rightmost",
    "frontmost": "frontmost",
    "backmost": "backmost",
    "corner": "corner",
    "in_corner": "corner",
    "center_of_room": "center_of_room",
    "middle_of_room": "center_of_room",
    "against_wall": "against_wall",
    "on_wall": "on_wall",
    "wall_mounted": "on_wall",
    "largest": "largest",
    "biggest": "largest",
    "largest_one": "largest",
    "smallest": "smallest",
    "smallest_one": "smallest",
    "tallest": "tallest",
    "shortest": "shortest",
    "group_of_two": "group_of_two",
    "group_of_three": "group_of_three",
    "middle_of_group_of_three": "middle_of_group_of_three",
    "end_of_row": "end_of_row",
    "open": "open",
    "closed": "closed",
    "empty": "empty",
}

COLOR_NAME_TO_RGB = {
    "black": (0.05, 0.05, 0.05),
    "white": (0.95, 0.95, 0.95),
    "gray": (0.55, 0.55, 0.55),
    "grey": (0.55, 0.55, 0.55),
    "red": (0.90, 0.15, 0.15),
    "green": (0.20, 0.70, 0.20),
    "blue": (0.15, 0.35, 0.90),
    "yellow": (0.95, 0.90, 0.20),
    "orange": (0.95, 0.55, 0.15),
    "brown": (0.55, 0.35, 0.20),
    "pink": (0.95, 0.60, 0.75),
    "purple": (0.55, 0.30, 0.75),
    "silver": (0.75, 0.75, 0.78),
}

TEXT_PROMPT_TEMPLATES = (
    "{}",
    "an indoor object called {}",
    "a 3d indoor object called {}",
)


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return " ".join(text.split())


def _normalize_relation_name(value: Any) -> str:
    text = _normalize_label(value).replace(" ", "_")
    return RELATION_ALIASES.get(text, text)


def _normalize_attribute_name(value: Any) -> str:
    text = _normalize_label(value).replace(" ", "_")
    if text in COLOR_NAME_TO_RGB:
        return text
    return UNARY_ATTR_ALIASES.get(text, text)


def _safe_sigmoid(value: float, scale: float) -> float:
    scale = max(abs(scale), EPS)
    scaled = max(min(float(value) / scale, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-scaled))


def _safe_exp_neg(value: float, scale: float) -> float:
    scale = max(abs(scale), EPS)
    return math.exp(-max(float(value), 0.0) / scale)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _geometric_mean(scores: Sequence[float], weights: Sequence[float]) -> float:
    if not scores:
        return 0.0
    total_weight = 0.0
    total_log = 0.0
    for score, weight in zip(scores, weights):
        w = max(float(weight), 0.0)
        total_weight += w
        total_log += w * math.log(max(float(score), 1e-6))
    if total_weight <= EPS:
        return 0.0
    return math.exp(total_log / total_weight)


def _soft_rank_score(values: Sequence[float], target_index: int, lower_is_better: bool, scale: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    order = np.argsort(arr if lower_is_better else -arr)
    rank_position = int(np.where(order == target_index)[0][0])
    if len(arr) == 1:
        rank_component = 1.0
    else:
        rank_component = 1.0 - (rank_position / (len(arr) - 1))
    best_value = float(arr[order[0]])
    gap_component = _safe_exp_neg(abs(float(arr[target_index]) - best_value), scale)
    return _clamp01(0.6 * rank_component + 0.4 * gap_component)


def _mean_pairwise_distance(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            total += float(np.linalg.norm(points[i] - points[j]))
            count += 1
    return total / max(count, 1)


def _point_segment_distance_xy(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p2 = point[:2]
    a2 = a[:2]
    b2 = b[:2]
    ab = b2 - a2
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        return float(np.linalg.norm(p2 - a2))
    t = float(np.dot(p2 - a2, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a2 + t * ab
    return float(np.linalg.norm(p2 - proj))


def _xy_iou(box_min_a: np.ndarray, box_max_a: np.ndarray, box_min_b: np.ndarray, box_max_b: np.ndarray) -> float:
    inter_min = np.maximum(box_min_a[:2], box_min_b[:2])
    inter_max = np.minimum(box_max_a[:2], box_max_b[:2])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_area = float(inter_size[0] * inter_size[1])
    area_a = float(np.prod(np.maximum(box_max_a[:2] - box_min_a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(box_max_b[:2] - box_min_b[:2], 0.0)))
    union = area_a + area_b - inter_area
    if union <= EPS:
        return 0.0
    return inter_area / union


def _axis_from_viewer_direction(direction: str) -> Tuple[np.ndarray, np.ndarray]:
    if direction == "+x":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if direction == "-x":
        return np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    if direction == "+y":
        return np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    if direction == "-y":
        return np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    raise ValueError(f"Unsupported viewer direction: {direction}")


def _normalize_vec3(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _bbox_from_center_size(center: Any, size: Any) -> Optional[Tuple[List[float], List[float]]]:
    center = _normalize_vec3(center)
    size = _normalize_vec3(size)
    if center is None or size is None:
        return None
    return (
        [c - s / 2.0 for c, s in zip(center, size)],
        [c + s / 2.0 for c, s in zip(center, size)],
    )


def _bbox_from_min_max(box_min: Any, box_max: Any) -> Optional[Tuple[List[float], List[float]]]:
    box_min = _normalize_vec3(box_min)
    box_max = _normalize_vec3(box_max)
    if box_min is None or box_max is None:
        return None
    return box_min, box_max


def _center_from_box(box: Optional[Tuple[List[float], List[float]]]) -> Optional[List[float]]:
    if box is None:
        return None
    b_min, b_max = box
    return [(b_min[i] + b_max[i]) / 2.0 for i in range(3)]


def _pred_center_from_record(pred_rec: Optional[Mapping[str, Any]]) -> Optional[List[float]]:
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


def _compute_distance_for_prediction(
    gt_rec: Mapping[str, Any],
    pred_map: Mapping[int, Mapping[str, Any]],
    pred_id: Optional[int],
) -> Optional[float]:
    gt_center = _normalize_vec3(gt_rec.get("center"))
    pred_rec = pred_map.get(pred_id) if pred_id is not None else None
    pred_center = _pred_center_from_record(pred_rec)
    if gt_center is None or pred_center is None:
        return None
    dx = gt_center[0] - pred_center[0]
    dy = gt_center[1] - pred_center[1]
    dz = gt_center[2] - pred_center[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _norm_class_name(text: Any) -> str:
    """类别名归一化（优先用 match_nr3d_gt_to_pred 的增强版，失败则用本地版）。"""
    if _normalize_class is not None:
        return _normalize_class(text)
    return _normalize_label(text)


def _class_matches(pred_class: Any, gt_class: Any) -> bool:
    """类别一致：归一化相等，或落在同一同义词组内（如 cabinet/kitchen cabinet）。"""
    a = _norm_class_name(pred_class)
    b = _norm_class_name(gt_class)
    if not a or not b:
        return False
    if a == b:
        return True
    for group in _SYNONYM_GROUPS:
        if a in group and b in group:
            return True
    return False


def _corresponds(pred_rec: Mapping[str, Any], gt_rec: Mapping[str, Any], thresh: float = 0.5) -> bool:
    """对应规则（用于 SelAcc / Coverage）：类别一致 且 3D 质心距离 <= thresh（米）。"""
    if not _class_matches(pred_rec.get("class_name"), gt_rec.get("label")):
        return False
    pred_map = {int(pred_rec["pred_id"]): pred_rec}
    d = _compute_distance_for_prediction(gt_rec, pred_map, int(pred_rec["pred_id"]))
    return d is not None and d <= thresh


def _summarize_dists(dists: Sequence[Optional[float]]) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], int, int]]:
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


def _render_progress(current: int, total: int, processed: int, skipped: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} processed={processed} skipped={skipped}"


def _make_progress_printer(width: int = PROGRESS_BAR_WIDTH):
    last_len = 0

    def _print_progress(current: int, total: int, processed: int, skipped: int) -> None:
        nonlocal last_len
        msg = _render_progress(current, total, processed, skipped, width=width)
        pad = " " * max(0, last_len - len(msg))
        print("\r" + msg + pad, end="", flush=True)
        last_len = len(msg)

    return _print_progress


def parse_response_json(text: str) -> Optional[Dict[str, Any]]:
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


@dataclass(frozen=True)
class QueryNode:
    node_id: str
    role: str
    class_name: Optional[str]
    attributes: Tuple[str, ...] = ()
    mention: Optional[str] = None


@dataclass(frozen=True)
class QueryRelation:
    rel_type: str
    source: str
    targets: Tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class QueryGraph:
    scene_id: Optional[str]
    viewer_direction: Optional[str]
    parser_confidence: Optional[float]
    reason: Optional[str]
    nodes: Tuple[QueryNode, ...]
    relations: Tuple[QueryRelation, ...]

    def target_node(self) -> QueryNode:
        for node in self.nodes:
            if node.role == "target":
                return node
        raise ValueError("Query graph is missing a target node")

    def node_map(self) -> Dict[str, QueryNode]:
        return {node.node_id: node for node in self.nodes}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QueryGraph":
        if payload.get("nodes") and payload.get("relations") is not None:
            return cls._from_nodes_and_relations(payload)
        if payload.get("parsed_chain"):
            return cls._from_parsed_chain(payload)
        raise ValueError("Response must contain either `nodes`/`relations` or `parsed_chain`.")

    @classmethod
    def _from_nodes_and_relations(cls, payload: Mapping[str, Any]) -> "QueryGraph":
        nodes: List[QueryNode] = []
        raw_nodes = payload.get("nodes") or []
        role_counts: Dict[str, int] = defaultdict(int)
        for raw_node in raw_nodes:
            role = _normalize_label(raw_node.get("role"))
            if role not in {"target", "ref"}:
                continue
            role_counts[role] += 1
            default_id = "t" if role == "target" else f"r{role_counts[role]}"
            attrs = tuple(
                _normalize_attribute_name(attr)
                for attr in (raw_node.get("attributes") or raw_node.get("attrs") or [])
                if _normalize_attribute_name(attr)
            )
            nodes.append(
                QueryNode(
                    node_id=str(raw_node.get("id") or default_id),
                    role=role,
                    class_name=_normalize_label(raw_node.get("class")),
                    attributes=attrs,
                    mention=raw_node.get("mention"),
                )
            )
        relations: List[QueryRelation] = []
        for raw_rel in payload.get("relations") or []:
            rel_type = _normalize_relation_name(raw_rel.get("type"))
            source = str(raw_rel.get("source") or raw_rel.get("from") or "")
            targets: List[str] = []
            target = raw_rel.get("target") or raw_rel.get("to")
            if target:
                targets.append(str(target))
            for value in raw_rel.get("targets") or []:
                targets.append(str(value))
            aux_target = raw_rel.get("aux_target")
            if aux_target:
                targets.append(str(aux_target))
            if rel_type and source and targets:
                relations.append(
                    QueryRelation(
                        rel_type=rel_type,
                        source=source,
                        targets=tuple(targets),
                        weight=float(raw_rel.get("weight", 1.0)),
                    )
                )
        return cls(
            scene_id=payload.get("scene_id"),
            viewer_direction=_normalize_viewer_direction(payload.get("viewer_direction")),
            parser_confidence=_coerce_float(payload.get("parser_confidence")),
            reason=payload.get("reason"),
            nodes=tuple(nodes),
            relations=tuple(relations),
        )

    @classmethod
    def _from_parsed_chain(cls, payload: Mapping[str, Any]) -> "QueryGraph":
        nodes: List[QueryNode] = []
        relations: List[QueryRelation] = []
        refs: List[QueryNode] = []
        target_node: Optional[QueryNode] = None
        for idx, raw_node in enumerate(payload.get("parsed_chain") or [], 1):
            role = _normalize_label(raw_node.get("role"))
            if role == "ref":
                node = QueryNode(
                    node_id=f"r{len(refs) + 1}",
                    role="ref",
                    class_name=_normalize_label(raw_node.get("class")),
                    attributes=tuple(
                        _normalize_attribute_name(attr)
                        for attr in (raw_node.get("constraints") or [])
                        if _is_unary_attribute(attr)
                    ),
                )
                refs.append(node)
                nodes.append(node)
            elif role == "target" and target_node is None:
                unary_attrs: List[str] = []
                for constraint in raw_node.get("constraints") or []:
                    compiled = _compile_constraint_to_relation(_normalize_label(constraint), refs)
                    if compiled is None:
                        unary_attrs.append(_normalize_attribute_name(constraint))
                    elif isinstance(compiled, QueryRelation):
                        relations.append(compiled)
                    else:
                        unary_attrs.append(compiled)
                target_node = QueryNode(
                    node_id="t",
                    role="target",
                    class_name=_normalize_label(raw_node.get("class")),
                    attributes=tuple(attr for attr in unary_attrs if attr),
                )
                nodes.append(target_node)
        if target_node is None:
            raise ValueError("parsed_chain must contain one target node")
        return cls(
            scene_id=payload.get("scene_id"),
            viewer_direction=_normalize_viewer_direction(payload.get("viewer_direction")),
            parser_confidence=_coerce_float(payload.get("parser_confidence")),
            reason=payload.get("reason"),
            nodes=tuple(nodes),
            relations=tuple(relations),
        )


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_viewer_direction(value: Any) -> Optional[str]:
    text = _normalize_label(value)
    if text in {"", "none", "null"}:
        return None
    if text in ALL_VIEWER_DIRECTIONS:
        return text
    return None


def _is_unary_attribute(value: Any) -> bool:
    text = _normalize_attribute_name(value)
    return text in UNARY_ATTR_ALIASES.values() or text in COLOR_NAME_TO_RGB


def _compile_constraint_to_relation(text: str, refs: Sequence[QueryNode]) -> Optional[Any]:
    if not text:
        return None
    attr_name = _normalize_attribute_name(text)
    if _is_unary_attribute(attr_name):
        return attr_name

    relation_specs = [
        ("left of", "left_of"),
        ("right of", "right_of"),
        ("in front of", "front_of"),
        ("front of", "front_of"),
        ("behind", "behind"),
        ("next to", "near"),
        ("near", "near"),
        ("closest to", "closest_to"),
        ("nearest to", "closest_to"),
        ("farthest from", "farthest_from"),
        ("furthest from", "farthest_from"),
        ("above", "above"),
        ("below", "below"),
        ("under", "under"),
        ("on top of", "on_top_of"),
        ("with", "with_object"),
        ("side of room with", "same_side_as"),
    ]
    for prefix, rel_type in relation_specs:
        if text.startswith(prefix):
            ref_id = _resolve_ref_id(text[len(prefix) :].strip(), refs)
            if ref_id is None:
                return None
            return QueryRelation(rel_type=rel_type, source="t", targets=(ref_id,), weight=1.0)
    if text.startswith("between") and len(refs) >= 2:
        ref_ids = _resolve_between_ref_ids(text, refs)
        if len(ref_ids) == 2:
            return QueryRelation(rel_type="between", source="t", targets=tuple(ref_ids), weight=1.0)
    return None


def _resolve_ref_id(text: str, refs: Sequence[QueryNode]) -> Optional[str]:
    clean = _normalize_label(text)
    if not refs:
        return None
    if not clean:
        return refs[0].node_id
    match = re.search(r"\bref(\d+)?\b", clean)
    if match:
        index = int(match.group(1) or "1") - 1
        index = max(0, min(index, len(refs) - 1))
        return refs[index].node_id
    for ref in refs:
        if ref.class_name and ref.class_name in clean:
            return ref.node_id
    return refs[0].node_id


def _resolve_between_ref_ids(text: str, refs: Sequence[QueryNode]) -> List[str]:
    matches = list(re.finditer(r"\bref(\d+)?\b", text))
    if len(matches) >= 2:
        resolved: List[str] = []
        for match in matches[:2]:
            index = int(match.group(1) or "1") - 1
            index = max(0, min(index, len(refs) - 1))
            resolved.append(refs[index].node_id)
        return resolved
    if len(refs) >= 2:
        return [refs[0].node_id, refs[1].node_id]
    return []


class CLIPTextMatcher:
    def __init__(self, model_name: str, pretrained: str, device: str):
        if torch is None:
            raise RuntimeError("The `torch` package is required to run CLIP-based label matching.")
        if open_clip is None:
            raise RuntimeError("The `open_clip_torch` package is required to run this script.")
        resolved_pretrained = pretrained if os.path.exists(pretrained) else pretrained
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=resolved_pretrained)
        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}

    def _encode_batch(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        missing = [text for text in texts if text not in self._cache]
        if missing:
            prompts: List[str] = []
            for text in missing:
                for template in TEXT_PROMPT_TEMPLATES:
                    prompts.append(template.format(text))
            with torch.no_grad():
                tokenized = self.tokenizer(prompts).to(self.device)
                feats = self.model.encode_text(tokenized)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)
            chunks = feats.reshape(len(missing), len(TEXT_PROMPT_TEMPLATES), -1)
            averaged = chunks.mean(dim=1)
            averaged = averaged / (averaged.norm(dim=-1, keepdim=True) + 1e-7)
            for text, feat in zip(missing, averaged):
                self._cache[text] = feat.detach().cpu()
        return {text: self._cache[text] for text in texts}

    def top_matches(self, query_text: str, candidate_texts: Sequence[str], top_k: int = 3) -> List[Tuple[str, float]]:
        query_text = _normalize_label(query_text)
        labels = [_normalize_label(label) for label in candidate_texts if _normalize_label(label)]
        if not query_text or not labels:
            return []
        embeddings = self._encode_batch([query_text, *labels])
        query_feat = embeddings[query_text]
        label_feats = torch.stack([embeddings[label] for label in labels], dim=0)
        sims = torch.mv(label_feats, query_feat)
        order = torch.argsort(sims, descending=True)
        results: List[Tuple[str, float]] = []
        for idx in order[: max(int(top_k), 1)]:
            label = labels[int(idx)]
            results.append((label, float(sims[int(idx)])))
        return results


@dataclass
class ResolvedClass:
    class_name: str
    similarity: float


@dataclass
class NodeCandidate:
    pred_id: int
    object_ref: SceneObject
    class_score: float
    unary_scores: Dict[str, float]
    unary_total: float
    unsupported_attributes: Tuple[str, ...]

    @property
    def prior_score(self) -> float:
        return self.unary_total


@dataclass
class NodeCandidatePool:
    node: QueryNode
    resolved_classes: Tuple[ResolvedClass, ...]
    full_objects: Tuple[SceneObject, ...]
    candidates: Tuple[NodeCandidate, ...]


@dataclass
class MatchState:
    assignment: Dict[str, NodeCandidate]
    log_sum: float
    weight_sum: float
    details: Dict[str, Any]
    # E10: raw (un-weighted-by-lambda) log accumulators of the node side and
    # the relation side, used for per-term normalization in the final ranking.
    node_log_sum: float = 0.0
    node_weight_sum: float = 0.0
    rel_log_sum: float = 0.0
    rel_weight_sum: float = 0.0

    @property
    def total_score(self) -> float:
        if self.weight_sum <= EPS:
            return 0.0
        return math.exp(self.log_sum / self.weight_sum)

    @property
    def node_part(self) -> Optional[float]:
        """Node-side geometric mean of the (role-weighted) node scores."""
        if self.node_weight_sum <= EPS:
            return None
        return math.exp(self.node_log_sum / self.node_weight_sum)

    @property
    def rel_part(self) -> Optional[float]:
        """Relation-side geometric mean of the (rel.weight-weighted) relation scores."""
        if self.rel_weight_sum <= EPS:
            return None
        return math.exp(self.rel_log_sum / self.rel_weight_sum)


class QueryGraphGrounder:
    def __init__(
        self,
        scene_graph: SpatialSceneGraph,
        clip_matcher: CLIPTextMatcher,
        *,
        target_limit: int = 24,
        ref_limit: int = 10,
        beam_size: int = 160,
        class_top_k: int = 3,
        min_clip_similarity: float = 0.18,
        class_margin: float = 0.08,
        disabled_scorers: Sequence[str] = (),
        graph_balance: Optional[float] = None,
        normalize_terms: bool = False,
    ):
        self.scene = scene_graph
        self.matcher = clip_matcher
        self.target_limit = max(int(target_limit), 1)
        self.ref_limit = max(int(ref_limit), 1)
        self.beam_size = max(int(beam_size), 1)
        self.class_top_k = max(int(class_top_k), 1)
        self.min_clip_similarity = float(min_clip_similarity)
        self.class_margin = float(class_margin)
        # E10: lambda_u — node-side share of the total weight in [0, 1];
        # relation side is 1 - lambda_u. None keeps the module-level default
        # (GRAPH_BALANCE), so existing runs (E2/E4/Table II) are unchanged.
        self.graph_balance = GRAPH_BALANCE if graph_balance is None else float(graph_balance)
        # E10: per-term normalization of the node vs relation log-terms before
        # ranking (addresses R1 "Eq.(9) terms are imbalanced"). Only affects
        # the ordering/beam pruning, never stored per-instance scores.
        self.normalize_terms = bool(normalize_terms)
        # which of the 7 relation scorers are ablated (their rel_types never
        # contribute to S(pi)); unknown names are ignored.
        self.disabled_scorers = frozenset(
            name for name in disabled_scorers if name in SCORER_RELTYPES
        )
        self.available_classes = sorted({_normalize_label(obj.class_name) for obj in self.scene.objects if _normalize_label(obj.class_name)})
        self.objects_by_class: Dict[str, List[SceneObject]] = defaultdict(list)
        for obj in self.scene.objects:
            self.objects_by_class[_normalize_label(obj.class_name)].append(obj)

    def _relation_enabled(self, rel_type: str) -> bool:
        return _RELTYPE_TO_SCORER.get(rel_type) not in self.disabled_scorers

    def ground(self, query_graph: QueryGraph, top_k: int = 5) -> Dict[str, Any]:
        node_map = query_graph.node_map()
        target_node = query_graph.target_node()
        directions = (query_graph.viewer_direction,) if query_graph.viewer_direction else ALL_VIEWER_DIRECTIONS
        best_by_target: Dict[int, Dict[str, Any]] = {}
        resolved_classes_by_node: Dict[str, List[ResolvedClass]] = {}

        for direction in directions:
            pools = {
                node.node_id: self._build_node_candidate_pool(node, direction)
                for node in query_graph.nodes
            }
            for node_id, pool in pools.items():
                resolved_classes_by_node[node_id] = list(pool.resolved_classes)
            states = self._search_assignments(query_graph, pools, direction)
            for state in states:
                target_candidate = state.assignment.get(target_node.node_id)
                if target_candidate is None:
                    continue
                pred_id = target_candidate.pred_id
                record = {
                    "pred_id": pred_id,
                    "class_name": target_candidate.object_ref.class_name,
                    "center": [float(x) for x in target_candidate.object_ref.center],
                    "size": [float(x) for x in target_candidate.object_ref.size],
                    "best_direction": direction,
                    "total_score": float(state.total_score),
                    "node_part": state.node_part,
                    "rel_part": state.rel_part,
                    "node_scores": state.details.get("node_scores", {}),
                    "relation_scores": state.details.get("relation_scores", {}),
                    "assignments": {
                        node_id: {
                            "pred_id": candidate.pred_id,
                            "class_name": candidate.object_ref.class_name,
                            "prior_score": candidate.prior_score,
                        }
                        for node_id, candidate in state.assignment.items()
                    },
                }
                current = best_by_target.get(pred_id)
                if current is None or record["total_score"] > current["total_score"]:
                    best_by_target[pred_id] = record

        results = sorted(best_by_target.values(), key=lambda item: item["total_score"], reverse=True)
        if self.normalize_terms and results:
            # E10 (R1: Eq.(9) node/relation terms are imbalanced): re-rank the
            # per-target winners by per-term normalized scores
            #   rank = lambda_u * norm(node_part) + (1 - lambda_u) * norm(rel_part)
            # with min-max normalization carried out ACROSS candidates, so the
            # two terms contribute on a comparable scale and lambda_u (the
            # node-relation balance) is a meaningful knob. Fall back to the raw
            # side that is present when the other side is absent (no relations,
            # or degenerate node scores).
            node_vals = [float(r["node_part"]) for r in results if r.get("node_part") is not None]
            rel_vals = [float(r["rel_part"]) for r in results if r.get("rel_part") is not None]
            lo_n, hi_n = (min(node_vals), max(node_vals)) if node_vals else (0.0, 1.0)
            lo_r, hi_r = (min(rel_vals), max(rel_vals)) if rel_vals else (0.0, 1.0)

            def norm(v, lo, hi):
                return 0.5 if hi - lo <= 1e-9 else (float(v) - lo) / (hi - lo)

            for r in results:
                n = r.get("node_part")
                rl = r.get("rel_part")
                if n is not None and rl is not None:
                    r["rank_score"] = (self.graph_balance * norm(n, lo_n, hi_n)
                                       + (1.0 - self.graph_balance) * norm(rl, lo_r, hi_r))
                elif n is not None:
                    r["rank_score"] = norm(n, lo_n, hi_n)
                else:
                    r["rank_score"] = norm(rl, lo_r, hi_r)
            results = sorted(results, key=lambda item: item["rank_score"], reverse=True)

        top_results = results[: max(int(top_k), 1)]
        return {
            "query_graph": _query_graph_to_dict(query_graph),
            "resolved_classes": {
                node_id: [vars(item) for item in matches]
                for node_id, matches in resolved_classes_by_node.items()
            },
            "room": self.scene.room_summary(),
            "results": top_results,
            "llm_context": self._format_results_for_llm(top_results),
        }

    def _build_node_candidate_pool(self, node: QueryNode, direction: str) -> NodeCandidatePool:
        resolved_classes = tuple(self._resolve_classes(node.class_name))
        full_objects: List[SceneObject] = []
        class_sim_by_name = {item.class_name: item.similarity for item in resolved_classes}
        if resolved_classes:
            for item in resolved_classes:
                full_objects.extend(self.objects_by_class.get(item.class_name, []))
        if not full_objects:
            full_objects = list(self.scene.objects)

        candidates: List[NodeCandidate] = []
        for obj in full_objects:
            class_sim = class_sim_by_name.get(_normalize_label(obj.class_name), self.min_clip_similarity)
            class_score = _clamp01((class_sim + 1.0) / 2.0)
            supported_scores: List[float] = [class_score]
            weights: List[float] = [2.0]
            unary_scores: Dict[str, float] = {"class": class_score}
            unsupported_attributes: List[str] = []
            for attr in node.attributes:
                score = self._score_unary_attribute(attr, obj, full_objects, direction)
                if score is None:
                    unsupported_attributes.append(attr)
                    continue
                unary_scores[attr] = score
                supported_scores.append(score)
                weights.append(1.0)
            total = _geometric_mean(supported_scores, weights)
            candidates.append(
                NodeCandidate(
                    pred_id=obj.pred_id,
                    object_ref=obj,
                    class_score=class_score,
                    unary_scores=unary_scores,
                    unary_total=total,
                    unsupported_attributes=tuple(unsupported_attributes),
                )
            )
        candidates.sort(key=lambda item: item.prior_score, reverse=True)
        limit = self.target_limit if node.role == "target" else self.ref_limit
        return NodeCandidatePool(
            node=node,
            resolved_classes=resolved_classes,
            full_objects=tuple(full_objects),
            candidates=tuple(candidates[:limit]),
        )

    def _resolve_classes(self, query_class: Optional[str]) -> List[ResolvedClass]:
        if not query_class or not self.available_classes:
            return []
        matches = self.matcher.top_matches(query_class, self.available_classes, top_k=self.class_top_k)
        if not matches:
            return []
        best_similarity = matches[0][1]
        resolved: List[ResolvedClass] = []
        for class_name, similarity in matches:
            if similarity < self.min_clip_similarity:
                continue
            if similarity + self.class_margin < best_similarity:
                continue
            resolved.append(ResolvedClass(class_name=class_name, similarity=similarity))
        if not resolved:
            class_name, similarity = matches[0]
            resolved.append(ResolvedClass(class_name=class_name, similarity=similarity))
        return resolved

    def _score_unary_attribute(
        self,
        attr: str,
        obj: SceneObject,
        peer_group: Sequence[SceneObject],
        direction: str,
    ) -> Optional[float]:
        attr = _normalize_attribute_name(attr)
        if attr in COLOR_NAME_TO_RGB:
            if obj.color_rgb is None:
                return None
            color_vec = np.asarray(COLOR_NAME_TO_RGB[attr], dtype=np.float64)
            pred_color = np.asarray(obj.color_rgb, dtype=np.float64)
            if pred_color.max() > 1.5:
                pred_color = pred_color / 255.0
            return _clamp01(1.0 - float(np.linalg.norm(pred_color - color_vec)) / math.sqrt(3.0))
        if attr == "upper":
            values = [item.center[2] for item in peer_group]
            return _soft_rank_score(values, self._find_index(peer_group, obj.pred_id), lower_is_better=False, scale=self.scene.vertical_scale)
        if attr == "lower":
            values = [item.center[2] for item in peer_group]
            return _soft_rank_score(values, self._find_index(peer_group, obj.pred_id), lower_is_better=True, scale=self.scene.vertical_scale)
        if attr == "largest":
            values = [item.volume for item in peer_group]
            return _soft_rank_score(values, self._find_index(peer_group, obj.pred_id), lower_is_better=False, scale=max(self.scene.horizontal_scale, 0.2))
        if attr == "smallest":
            values = [item.volume for item in peer_group]
            return _soft_rank_score(values, self._find_index(peer_group, obj.pred_id), lower_is_better=True, scale=max(self.scene.horizontal_scale, 0.2))
        if attr == "tallest":
            values = [float(item.size[2]) for item in peer_group]
            return _soft_rank_score(values, self._find_index(peer_group, obj.pred_id), lower_is_better=False, scale=self.scene.vertical_scale)
        if attr == "shortest":
            values = [float(item.size[2]) for item in peer_group]
            return _soft_rank_score(values, self._find_index(peer_group, obj.pred_id), lower_is_better=True, scale=self.scene.vertical_scale)
        if attr in {"leftmost", "rightmost", "frontmost", "backmost"}:
            front_axis, left_axis = _axis_from_viewer_direction(direction)
            centers = np.stack([item.center for item in peer_group], axis=0)
            index = self._find_index(peer_group, obj.pred_id)
            if attr == "leftmost":
                values = [float(np.dot(center, left_axis)) for center in centers]
                return _soft_rank_score(values, index, lower_is_better=False, scale=self.scene.align_scale)
            if attr == "rightmost":
                values = [float(np.dot(center, left_axis)) for center in centers]
                return _soft_rank_score(values, index, lower_is_better=True, scale=self.scene.align_scale)
            if attr == "frontmost":
                values = [float(np.dot(center, front_axis)) for center in centers]
                return _soft_rank_score(values, index, lower_is_better=False, scale=self.scene.align_scale)
            values = [float(np.dot(center, front_axis)) for center in centers]
            return _soft_rank_score(values, index, lower_is_better=True, scale=self.scene.align_scale)
        if attr == "corner":
            dist, _ = self.scene.nearest_corner_distance(obj)
            return _safe_exp_neg(dist, self.scene.corner_scale)
        if attr == "center_of_room":
            return _safe_exp_neg(float(np.linalg.norm(obj.center - self.scene.room_center)), self.scene.midpoint_scale)
        if attr == "against_wall":
            return max(self.scene.wall_affinity(obj).values())
        if attr == "on_wall":
            against = max(self.scene.wall_affinity(obj).values())
            clearance = float(max(obj.box_min[2] - self.scene.room_min[2], 0.0))
            elevated = 1.0 - _safe_exp_neg(clearance, self.scene.vertical_scale)
            return _clamp01(against * elevated)
        if attr == "group_of_two":
            score, _ = self._best_group_membership(obj, peer_group, count=2)
            return score
        if attr == "group_of_three":
            score, _ = self._best_group_membership(obj, peer_group, count=3)
            return score
        if attr == "middle_of_group_of_three":
            score, _ = self._best_middle_in_group(obj, peer_group, count=3)
            return score
        if attr == "end_of_row":
            score, _ = self._best_end_of_row(obj, peer_group, count=3)
            return score
        return None

    def _find_index(self, group: Sequence[SceneObject], pred_id: int) -> int:
        for idx, item in enumerate(group):
            if item.pred_id == pred_id:
                return idx
        raise ValueError(f"pred_id={pred_id} not found in candidate group")

    def _search_assignments(
        self,
        query_graph: QueryGraph,
        node_pools: Mapping[str, NodeCandidatePool],
        direction: str,
    ) -> List[MatchState]:
        ordered_nodes = sorted(query_graph.nodes, key=lambda node: (0 if node.role == "target" else 1, len(node_pools[node.node_id].candidates)))
        beam: List[MatchState] = [MatchState(assignment={}, log_sum=0.0, weight_sum=0.0, details={"node_scores": {}, "relation_scores": {}})]
        relation_list = list(query_graph.relations)

        for node in ordered_nodes:
            pool = node_pools[node.node_id]
            next_beam: List[MatchState] = []
            for state in beam:
                used_pred_ids = {candidate.pred_id for candidate in state.assignment.values()}
                for candidate in pool.candidates:
                    if candidate.pred_id in used_pred_ids:
                        continue
                    log_sum = state.log_sum
                    weight_sum = state.weight_sum
                    node_log_sum = state.node_log_sum
                    node_weight_sum = state.node_weight_sum
                    rel_log_sum = state.rel_log_sum
                    rel_weight_sum = state.rel_weight_sum
                    details = {
                        "node_scores": dict(state.details.get("node_scores", {})),
                        "relation_scores": dict(state.details.get("relation_scores", {})),
                    }
                    new_assignment = dict(state.assignment)
                    new_assignment[node.node_id] = candidate

                    ########################################
                    node_weight = self.graph_balance * (
                        TARGET_NODE_ROLE_WEIGHT if node.role == "target" else REF_NODE_ROLE_WEIGHT
                    )
                    node_raw_w = TARGET_NODE_ROLE_WEIGHT if node.role == "target" else REF_NODE_ROLE_WEIGHT
                    node_score = max(candidate.prior_score, 1e-6)
                    ##############################################
                    log_sum += node_weight * math.log(node_score)
                    weight_sum += node_weight
                    node_log_sum += node_raw_w * math.log(node_score)
                    node_weight_sum += node_raw_w
                    details["node_scores"][node.node_id] = {
                        "pred_id": candidate.pred_id,
                        "score": candidate.prior_score,
                        "components": candidate.unary_scores,
                        "unsupported_attributes": list(candidate.unsupported_attributes),
                    }

                    for rel_idx, relation in enumerate(relation_list):
                        if not self._relation_enabled(relation.rel_type):
                            continue  # ablated scorer: relation does not contribute
                        rel_key = f"{rel_idx}:{relation.rel_type}:{relation.source}->{','.join(relation.targets)}"
                        if rel_key in details["relation_scores"]:
                            continue
                        if relation.source not in new_assignment:
                            continue
                        if any(target_id not in new_assignment for target_id in relation.targets):
                            continue
                        relation_score = self._score_relation(relation, new_assignment, node_pools, direction)
                        relation_score = max(relation_score, 1e-6)
                        effective_relation_weight = (1.0 - self.graph_balance) * relation.weight
                        log_sum += effective_relation_weight * math.log(relation_score)
                        weight_sum += effective_relation_weight
                        rel_log_sum += relation.weight * math.log(relation_score)
                        rel_weight_sum += relation.weight
                        details["relation_scores"][rel_key] = {
                            "type": relation.rel_type,
                            "score": relation_score,
                            "weight": relation.weight,
                            "effective_weight": effective_relation_weight,
                            "source": relation.source,
                            "targets": list(relation.targets),
                        }
                    next_beam.append(MatchState(assignment=new_assignment, log_sum=log_sum, weight_sum=weight_sum,
                                                details=details,
                                                node_log_sum=node_log_sum, node_weight_sum=node_weight_sum,
                                                rel_log_sum=rel_log_sum, rel_weight_sum=rel_weight_sum))
            next_beam.sort(key=lambda item: item.total_score, reverse=True)
            beam = next_beam[: self.beam_size]
            if not beam:
                break
        return beam

    def _score_relation(
        self,
        relation: QueryRelation,
        assignment: Mapping[str, NodeCandidate],
        node_pools: Mapping[str, NodeCandidatePool],
        direction: str,
    ) -> float:
        source_obj = assignment[relation.source].object_ref
        target_objs = [assignment[target_id].object_ref for target_id in relation.targets]
        rel_type = relation.rel_type

        if rel_type == "between" and len(target_objs) >= 2:
            return self._score_between(source_obj, target_objs[0], target_objs[1])
        if len(target_objs) != 1:
            return 0.0
        target_obj = target_objs[0]

        if rel_type == "left_of":
            return self._score_left_right(source_obj, target_obj, direction, want="left")
        if rel_type == "right_of":
            return self._score_left_right(source_obj, target_obj, direction, want="right")
        if rel_type == "front_of":
            return self._score_front_back(source_obj, target_obj, direction, want="front")
        if rel_type == "behind":
            return self._score_front_back(source_obj, target_obj, direction, want="back")
        if rel_type == "near":
            return self._score_near(source_obj, target_obj)
        if rel_type == "closest_to":
            pool = node_pools[relation.source].full_objects
            values = [self._distance_3d(obj, target_obj) for obj in pool]
            index = self._find_index(pool, source_obj.pred_id)
            return _soft_rank_score(values, index, lower_is_better=True, scale=self.scene.horizontal_scale)
        if rel_type == "farthest_from":
            pool = node_pools[relation.source].full_objects
            values = [self._distance_3d(obj, target_obj) for obj in pool]
            index = self._find_index(pool, source_obj.pred_id)
            return _soft_rank_score(values, index, lower_is_better=False, scale=self.scene.horizontal_scale)
        if rel_type == "above":
            return self._score_above(source_obj, target_obj, require_contact=False)
        if rel_type == "below":
            return self._score_below(source_obj, target_obj, require_contact=False)
        if rel_type == "under":
            return self._score_below(source_obj, target_obj, require_contact=True)
        if rel_type == "on_top_of":
            return self._score_above(source_obj, target_obj, require_contact=True)
        if rel_type == "same_side_as":
            return self._score_same_side(source_obj, target_obj)
        if rel_type == "with_object":
            return self._score_near(source_obj, target_obj)
        if rel_type == "has_on_it":
            return self._score_above(target_obj, source_obj, require_contact=True)
        return 0.0

    def _distance_3d(self, a: SceneObject, b: SceneObject) -> float:
        return float(np.linalg.norm(a.center - b.center))

    def _score_left_right(self, candidate: SceneObject, ref: SceneObject, direction: str, want: str) -> float:
        _, left_axis = _axis_from_viewer_direction(direction)
        front_axis, _ = _axis_from_viewer_direction(direction)
        delta = candidate.center - ref.center
        left_proj = float(np.dot(delta, left_axis))
        front_proj = float(np.dot(delta, front_axis))
        sign = 1.0 if want == "left" else -1.0
        dir_score = _safe_sigmoid(sign * left_proj, self.scene.horizontal_scale)
        orth_penalty = _safe_exp_neg(abs(front_proj), self.scene.horizontal_scale * 1.5)
        return _clamp01(dir_score * orth_penalty)

    def _score_front_back(self, candidate: SceneObject, ref: SceneObject, direction: str, want: str) -> float:
        front_axis, left_axis = _axis_from_viewer_direction(direction)
        delta = candidate.center - ref.center
        front_proj = float(np.dot(delta, front_axis))
        left_proj = float(np.dot(delta, left_axis))
        sign = 1.0 if want == "front" else -1.0
        dir_score = _safe_sigmoid(sign * front_proj, self.scene.horizontal_scale)
        orth_penalty = _safe_exp_neg(abs(left_proj), self.scene.horizontal_scale * 1.5)
        return _clamp01(dir_score * orth_penalty)

    def _score_near(self, candidate: SceneObject, ref: SceneObject) -> float:
        edge = self.scene.object_edge(candidate.pred_id, ref.pred_id)
        distance_xy = float(edge["distance_xy"])
        overlap_bonus = max(float(edge["xy_iou"]), 0.2)
        return _clamp01(_safe_exp_neg(distance_xy, self.scene.horizontal_scale) * overlap_bonus / 0.2)

    def _score_above(self, candidate: SceneObject, ref: SceneObject, require_contact: bool) -> float:
        delta_z = float(candidate.center[2] - ref.center[2])
        direction = _safe_sigmoid(delta_z, self.scene.vertical_scale)
        xy_align = max(
            _xy_iou(candidate.box_min, candidate.box_max, ref.box_min, ref.box_max),
            _safe_exp_neg(float(np.linalg.norm(candidate.center[:2] - ref.center[:2])), self.scene.horizontal_scale),
        )
        if not require_contact:
            return _clamp01(direction * xy_align)
        vertical_gap = abs(float(candidate.box_min[2] - ref.box_max[2]))
        contact = _safe_exp_neg(vertical_gap, self.scene.vertical_scale)
        return _clamp01(direction * xy_align * contact)

    def _score_below(self, candidate: SceneObject, ref: SceneObject, require_contact: bool) -> float:
        delta_z = float(ref.center[2] - candidate.center[2])
        direction = _safe_sigmoid(delta_z, self.scene.vertical_scale)
        xy_align = max(
            _xy_iou(candidate.box_min, candidate.box_max, ref.box_min, ref.box_max),
            _safe_exp_neg(float(np.linalg.norm(candidate.center[:2] - ref.center[:2])), self.scene.horizontal_scale),
        )
        if not require_contact:
            return _clamp01(direction * xy_align)
        vertical_gap = abs(float(ref.box_min[2] - candidate.box_max[2]))
        contact = _safe_exp_neg(vertical_gap, self.scene.vertical_scale)
        return _clamp01(direction * xy_align * contact)

    def _score_same_side(self, candidate: SceneObject, ref: SceneObject) -> float:
        a = self.scene.wall_affinity(candidate)
        b = self.scene.wall_affinity(ref)
        return _clamp01(max(min(a[name], b[name]) for name in a))

    def _score_between(self, candidate: SceneObject, ref_a: SceneObject, ref_b: SceneObject) -> float:
        if candidate.pred_id in {ref_a.pred_id, ref_b.pred_id} or ref_a.pred_id == ref_b.pred_id:
            return 0.0
        midpoint = (ref_a.center + ref_b.center) / 2.0
        midpoint_score = _safe_exp_neg(float(np.linalg.norm(candidate.center - midpoint)), self.scene.midpoint_scale)
        balance_score = _safe_exp_neg(abs(self._distance_3d(candidate, ref_a) - self._distance_3d(candidate, ref_b)), self.scene.midpoint_scale)
        line_score = _safe_exp_neg(_point_segment_distance_xy(candidate.center, ref_a.center, ref_b.center), self.scene.align_scale)
        return _clamp01(midpoint_score * balance_score * line_score)

    def _local_same_class_group(self, candidate: SceneObject, peer_group: Sequence[SceneObject], count: int) -> List[SceneObject]:
        neighbors = [item for item in peer_group if item.pred_id != candidate.pred_id]
        neighbors.sort(key=lambda item: float(np.linalg.norm(item.center - candidate.center)))
        limit = max(count + 3, 6)
        return [candidate] + neighbors[: max(limit - 1, 0)]

    def _best_group_membership(self, candidate: SceneObject, peer_group: Sequence[SceneObject], count: int) -> Tuple[float, List[int]]:
        local_group = self._local_same_class_group(candidate, peer_group, count=count)
        if len(local_group) < count or count <= 0:
            return 0.0, []
        candidate_id = candidate.pred_id
        best_score = 0.0
        best_group: List[int] = []
        local_by_id = {item.pred_id: item for item in local_group}
        other_ids = [item.pred_id for item in local_group if item.pred_id != candidate_id]
        for subset in _combinations(other_ids, count - 1):
            group_ids = [candidate_id, *subset]
            points = np.stack([local_by_id[pred_id].center for pred_id in group_ids], axis=0)
            compactness = _safe_exp_neg(_mean_pairwise_distance(points), self.scene.group_scale)
            if compactness > best_score:
                best_score = compactness
                best_group = group_ids
        return best_score, best_group

    def _best_middle_in_group(self, candidate: SceneObject, peer_group: Sequence[SceneObject], count: int) -> Tuple[float, List[int]]:
        local_group = self._local_same_class_group(candidate, peer_group, count=count)
        if len(local_group) < count or count <= 1:
            return 0.0, []
        candidate_id = candidate.pred_id
        best_score = 0.0
        best_group: List[int] = []
        local_by_id = {item.pred_id: item for item in local_group}
        other_ids = [item.pred_id for item in local_group if item.pred_id != candidate_id]
        for subset in _combinations(other_ids, count - 1):
            group_ids = [candidate_id, *subset]
            centers = np.stack([local_by_id[pred_id].center for pred_id in group_ids], axis=0)
            xy = centers[:, :2]
            compactness = _safe_exp_neg(_mean_pairwise_distance(centers), self.scene.group_scale)
            centered = xy - np.mean(xy, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            projections = xy @ axis
            candidate_position = int(group_ids.index(candidate_id))
            middle_index = len(projections) // 2
            order = np.argsort(projections)
            rank_position = int(np.where(order == candidate_position)[0][0])
            rank_gap = abs(rank_position - middle_index)
            middle_rank_score = 1.0 - (rank_gap / max(len(projections) - 1, 1))
            center_score = _safe_exp_neg(float(np.linalg.norm(candidate.center - np.mean(centers, axis=0))), self.scene.midpoint_scale)
            score = _clamp01(compactness * 0.5 * (middle_rank_score + center_score))
            if score > best_score:
                best_score = score
                best_group = group_ids
        return best_score, best_group

    def _best_end_of_row(self, candidate: SceneObject, peer_group: Sequence[SceneObject], count: int) -> Tuple[float, List[int]]:
        local_group = self._local_same_class_group(candidate, peer_group, count=count)
        if len(local_group) < count or count < 3:
            return 0.0, []
        candidate_id = candidate.pred_id
        best_score = 0.0
        best_group: List[int] = []
        local_by_id = {item.pred_id: item for item in local_group}
        other_ids = [item.pred_id for item in local_group if item.pred_id != candidate_id]
        for subset in _combinations(other_ids, count - 1):
            group_ids = [candidate_id, *subset]
            centers = np.stack([local_by_id[pred_id].center for pred_id in group_ids], axis=0)
            xy = centers[:, :2]
            centered = xy - np.mean(xy, axis=0, keepdims=True)
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
            denom = float(np.sum(singular_values)) + EPS
            collinearity = float(singular_values[0] / denom)
            axis = vh[0]
            projections = xy @ axis
            candidate_position = int(group_ids.index(candidate_id))
            rank_score = max(
                _soft_rank_score(projections, candidate_position, lower_is_better=True, scale=self.scene.align_scale),
                _soft_rank_score(projections, candidate_position, lower_is_better=False, scale=self.scene.align_scale),
            )
            compactness = _safe_exp_neg(_mean_pairwise_distance(centers), self.scene.group_scale * 1.5)
            score = _clamp01(collinearity * rank_score * compactness)
            if score > best_score:
                best_score = score
                best_group = group_ids
        return best_score, best_group

    def _format_results_for_llm(self, results: Sequence[Mapping[str, Any]]) -> str:
        lines: List[str] = []
        for result in results:
            parts = [
                f"pred_id={result['pred_id']}",
                f"class={result['class_name']}",
                f"total={result['total_score']:.3f}",
            ]
            if result.get("best_direction"):
                parts.append(f"dir={result['best_direction']}")
            for relation_key, relation_value in (result.get("relation_scores") or {}).items():
                _ = relation_key
                parts.append(f"{relation_value['type']}={relation_value['score']:.3f}")
            lines.append(", ".join(parts))
        return "\n".join(lines)


def _combinations(items: Sequence[int], choose: int) -> Iterable[Tuple[int, ...]]:
    if choose <= 0:
        yield ()
        return
    if len(items) < choose:
        return
    indices = list(range(choose))
    n = len(items)
    while True:
        yield tuple(items[i] for i in indices)
        for pos in reversed(range(choose)):
            if indices[pos] != pos + n - choose:
                break
        else:
            return
        indices[pos] += 1
        for nxt in range(pos + 1, choose):
            indices[nxt] = indices[nxt - 1] + 1


def _query_graph_to_dict(query_graph: QueryGraph) -> Dict[str, Any]:
    return {
        "scene_id": query_graph.scene_id,
        "viewer_direction": query_graph.viewer_direction,
        "parser_confidence": query_graph.parser_confidence,
        "reason": query_graph.reason,
        "nodes": [
            {
                "id": node.node_id,
                "role": node.role,
                "class": node.class_name,
                "attributes": list(node.attributes),
                "mention": node.mention,
            }
            for node in query_graph.nodes
        ],
        "relations": [
            {
                "type": relation.rel_type,
                "source": relation.source,
                "targets": list(relation.targets),
                "weight": relation.weight,
            }
            for relation in query_graph.relations
        ],
    }


def build_parse_prompt(gt_rec: Mapping[str, Any], preds: Sequence[Mapping[str, Any]]) -> str:
    scene = gt_rec["scene_id"]
    desc = gt_rec.get("description") or gt_rec.get("gt_description") or ""
    class_counts: Dict[str, int] = defaultdict(int)
    for pred in preds:
        class_name = _normalize_label(pred.get("class_name"))
        if class_name:
            class_counts[class_name] += 1
    class_inventory = [
        {"class_name": class_name, "count": count}
        for class_name, count in sorted(class_counts.items())
    ]
    return f"""You are a 3D referring-expression parser for a downstream query-graph grounder.
    Your task is ONLY to convert the language description into a compact structured query graph.
    Do NOT choose a predicted object id.

    Available candidate class inventory for this scene:
    {json.dumps(class_inventory, ensure_ascii=False, indent=2)}

    Output JSON schema:
    {{
    "scene_id": "{scene}",
    "viewer_direction": "+x/-x/+y/-y or null",
    "parser_confidence": 0-1,
    "reason": "...",
    "nodes": [
        {{"id": "t", "role": "target", "class": "...", "attributes": ["..."]}},
        {{"id": "r1", "role": "ref", "class": "...", "attributes": ["..."]}}
    ],
    "relations": [
        {{"type": "right_of", "source": "t", "target": "r1", "weight": 1.0}}
    ]
    }}

    Rules:
    1) There must be exactly one target node with id="t".
    2) Add reference nodes r1, r2, ... only when needed.
    3) Use a closed relation vocabulary only:
    left_of, right_of, front_of, behind, near, closest_to, farthest_from,
    above, below, under, on_top_of, between, same_side_as, with_object, has_on_it
    4) Use node attributes for:
    upper, lower, leftmost, rightmost, frontmost, backmost,
    corner, center_of_room, against_wall, on_wall,
    largest, smallest, tallest, shortest,
    group_of_two, group_of_three, middle_of_group_of_three, end_of_row,
    open, closed, empty,
    red, blue, green, black, white, brown, gray, yellow, orange, pink, purple
    5) If a relation is not in the vocabulary, approximate it with the closest supported relation instead of inventing a new name.
    6) For "between", use this format:
    {{"type": "between", "source": "t", "targets": ["r1", "r2"], "weight": 1.0}}
    7) viewer_direction should only be set when viewer-centric terms such as left/right/front/back need it.
    8) Keep reason short, at most 2 short sentences.
    9) Output JSON only.

    Example:
    Description: "the middle of three upper cabinets to the right of the refrigerator."
    Example output:
    {{
    "scene_id": "scene0000_00",
    "viewer_direction": "+x",
    "parser_confidence": 0.92,
    "reason": "target cabinet is upper and right of the refrigerator, and is the middle one among three cabinets.",
    "nodes": [
        {{"id": "t", "role": "target", "class": "cabinet", "attributes": ["upper", "middle_of_group_of_three"]}},
        {{"id": "r1", "role": "ref", "class": "refrigerator", "attributes": []}}
    ],
    "relations": [
        {{"type": "right_of", "source": "t", "target": "r1", "weight": 1.0}}
    ]
    }}

    Target description:
    "{desc}"
"""


def build_rerank_prompt(
    description: str,
    query_graph: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    compact = []
    for candidate in candidates:
        compact.append(
            {
                "pred_id": candidate.get("pred_id"),
                "class_name": candidate.get("class_name"),
                "best_direction": candidate.get("best_direction"),
                "total_score": candidate.get("total_score"),
                "assignments": candidate.get("assignments"),
                "node_scores": candidate.get("node_scores"),
                "relation_scores": candidate.get("relation_scores"),
            }
        )
    return f"""You are selecting the final target instance from a shortlist produced by a spatial graph matcher.
Use the original description together with the structured query graph and candidate evidence.
Choose one pred_id from the shortlist only.

Description:
"{description}"

Query graph:
{json.dumps(query_graph, ensure_ascii=False, indent=2)}

Shortlist:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Output JSON only:
{{
  "best_pred_id": <one pred_id from the shortlist>,
  "confidence": 0-1,
  "reason": "..."
}}
"""


def choose_reranked_pred_id(
    client: OpenAI,
    model: str,
    description: str,
    query_graph: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[str]]:
    if not candidates:
        return None, None, None
    prompt = build_rerank_prompt(description, query_graph, candidates)
    answer = request_llm_json_text(client, model, prompt, max_tokens=256)
    parsed = parse_response_json(answer)
    if not isinstance(parsed, dict):
        return None, None, answer.strip() or None
    pred_id = parsed.get("best_pred_id")
    try:
        pred_id = int(pred_id)
    except (TypeError, ValueError):
        pred_id = None
    shortlist_ids = {int(item["pred_id"]) for item in candidates if item.get("pred_id") is not None}
    if pred_id not in shortlist_ids:
        return None, parsed, answer.strip() or None
    return pred_id, parsed, answer.strip() or None


def _load_grounding_config(path: str) -> Dict[str, Any]:
    """Load the YAML eval config into a flat {arg_name: value} dict.
    Only keys present in the YAML are returned; CLI args override them."""
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to load config {path}: {exc}")
        return {}

    flat: Dict[str, Any] = {}
    l = cfg.get("llm") or {}
    if l.get("base_url"):
        flat["openai_base_url"] = l["base_url"]
    if l.get("api_key"):
        flat["openai_api_key"] = l["api_key"]
    if l.get("model"):
        flat["llm_model"] = l["model"]

    d = cfg.get("data") or {}
    if d.get("gt_path"):
        flat["gt_path"] = d["gt_path"]
    if d.get("oas_root"):
        flat["oas_root"] = d["oas_root"]
    if d.get("pred_name"):
        flat["pred_name"] = d["pred_name"]

    o = cfg.get("output") or {}
    if o.get("out"):
        flat["out"] = Path(o["out"])
    if o.get("metrics_out"):
        flat["metrics_out"] = Path(o["metrics_out"])

    e = cfg.get("eval") or {}
    for k, arg in [("scene", "scene"), ("max", "max"), ("print_every", "print_every"),
                   ("corr_thresh", "corr_thresh")]:
        if e.get(k) is not None:
            flat[arg] = e[k]

    g = cfg.get("grounder") or {}
    for k, arg in [("top_k", "top_k"), ("rerank_top_k", "rerank_top_k"),
                   ("disable_rerank", "disable_rerank"), ("beam_size", "beam_size"),
                   ("target_cands", "target_cands"), ("ref_cands", "ref_cands"),
                   ("class_top_k", "class_top_k")]:
        if g.get(k) is not None:
            flat[arg] = g[k]

    c = cfg.get("clip") or {}
    if c.get("model"):
        flat["clip_model"] = c["model"]
    if c.get("pretrained"):
        flat["clip_pretrained"] = c["pretrained"]
    if c.get("device"):
        flat["clip_device"] = c["device"]

    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM query-graph grounding with CLIP-based class matching.")
    parser.add_argument("--scene", default=None, help="Only process a single scene_id")
    parser.add_argument("--max", type=int, default=None, help="Max number of GT descriptions to process")
    parser.add_argument("--out", type=Path, default=Path("output_test/nr3d/qwen_outputs_y.json"))
    parser.add_argument("--metrics-out", type=Path, default=Path("output_test/nr3d/llm_query_graph_clip_metrics.json"))
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5, help="How many graph candidates to keep")
    parser.add_argument("--rerank-top-k", type=int, default=4, help="How many candidates to expose to LLM reranker")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable final LLM shortlist reranking")
    parser.add_argument(
        "--disable-scorer",
        default="",
        help="Comma-separated relation scorers to ablate from {%s}; e.g. 'dir,near'" % ",".join(SCORER_NAMES),
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-pretrained", default=str(DEFAULT_CLIP_PRETRAINED))
    default_clip_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser.add_argument("--clip-device", default=default_clip_device)
    parser.add_argument("--beam-size", type=int, default=160)
    parser.add_argument("--target-cands", type=int, default=24)
    parser.add_argument("--ref-cands", type=int, default=10)
    parser.add_argument("--class-top-k", type=int, default=3)
    parser.add_argument("--corr-thresh", type=float, default=0.5,
                        help="correspondence threshold (m) for SelAcc/Coverage: class match + centroid distance")
    parser.add_argument("--graph-balance", type=float, default=None,
                        help="lambda_u: node-side share of total matching weight in [0,1] "
                             "(None = module default GRAPH_BALANCE). Relation side = 1 - lambda_u.")
    parser.add_argument("--normalize-terms", action="store_true",
                        help="per-term normalize node vs relation scores before ranking "
                             "(E10; addresses R1 'Eq.(9) terms are imbalanced')")
    parser.add_argument("--config", type=str, default="config/grounding_eval.yaml",
                        help="YAML config overriding defaults; explicit CLI args take precedence")
    parser.add_argument("--gt-path", type=str, default=str(GT_PATH),
                        help="GT annotation JSON (NR3D/SR3D matched)")
    parser.add_argument("--oas-root", type=str, default=str(OAS_ROOT),
                        help="reconstruction instance root (contains sceneXXX/pred_bboxes.json)")
    parser.add_argument("--pred-name", type=str, default=PRED_NAME,
                        help="predicted bbox file name per scene")

    # pass 1: read --config (and any explicit CLI args)
    args = parser.parse_known_args()[0]
    flat = _load_grounding_config(args.config)
    if flat:
        parser.set_defaults(**flat)
    # pass 2: final parse — CLI args override config, config overrides defaults
    args = parser.parse_args()

    gts = json.loads(Path(args.gt_path).read_text())
    if args.scene:
        gts = [item for item in gts if item.get("scene_id") == args.scene]
    max_limit = MAX_GT if args.max is None else min(int(args.max), MAX_GT)
    if max_limit is not None:
        if len(gts) > max_limit:
            gts = random.sample(gts, k=max_limit)
        else:
            random.shuffle(gts)
    if not gts:
        raise SystemExit("没有匹配到可处理的 GT 描述记录")

    # per-scene unique GT objects (for Coverage / wrong-selection attribution)
    scene_gt_objs: Dict[str, Dict[Any, Mapping[str, Any]]] = {}
    for _g in gts:
        scene_gt_objs.setdefault(_g.get("scene_id"), {})[_g.get("object_id")] = _g

    client = make_client(args.openai_base_url, args.openai_api_key)
    clip_matcher = CLIPTextMatcher(args.clip_model, args.clip_pretrained, args.clip_device)

    pred_cache: Dict[str, List[Dict[str, Any]]] = {}
    pred_map_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
    scene_graph_cache: Dict[str, SpatialSceneGraph] = {}

    processed = 0
    skipped = 0
    total = len(gts)
    dists: List[Optional[float]] = []
    # grounding-protocol counters (Coverage / SelAcc / error split)
    n_covered = 0
    n_correct = 0
    error_cnt = {"not_reconstructed": 0, "no_prediction": 0, "wrong_selection": 0, "hallucinated": 0}
    progress = _make_progress_printer()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as out_f:
        for idx, gt_rec in enumerate(gts, 1):
            scene_id = gt_rec.get("scene_id")
            pred_path = Path(args.oas_root) / scene_id / args.pred_name
            if not pred_path.is_file():
                skipped += 1
                progress(idx, total, processed, skipped)
                continue

            if scene_id not in pred_cache:
                preds = json.loads(pred_path.read_text())
                pred_cache[scene_id] = preds
                pred_map_cache[scene_id] = {int(item["pred_id"]): item for item in preds if "pred_id" in item}
                scene_graph_cache[scene_id] = SpatialSceneGraph.from_prediction_records(preds)

            preds = pred_cache[scene_id]
            pred_map = pred_map_cache[scene_id]
            scene_graph = scene_graph_cache[scene_id]

            parse_prompt = build_parse_prompt(gt_rec, preds)
            parse_answer = request_llm_json_text(client, args.llm_model, parse_prompt, max_tokens=512)
            
            parse_payload = parse_response_json(parse_answer)
            query_graph: Optional[QueryGraph] = None
            parse_error: Optional[str] = None
            if isinstance(parse_payload, dict):
                try:
                    query_graph = QueryGraph.from_payload(parse_payload)
                except Exception as exc:  # noqa: BLE001
                    parse_error = str(exc)
            else:
                parse_error = "LLM parser did not return valid JSON"

            graph_result: Optional[Dict[str, Any]] = None
            graph_error: Optional[str] = None
            final_best_pred_id: Optional[int] = None
            rerank_payload: Optional[Dict[str, Any]] = None
            rerank_raw: Optional[str] = None

            if query_graph is not None:
                try:
                    grounder = QueryGraphGrounder(
                        scene_graph,
                        clip_matcher,
                        target_limit=args.target_cands,
                        ref_limit=args.ref_cands,
                        beam_size=args.beam_size,
                        class_top_k=args.class_top_k,
                        disabled_scorers=(
                            [s.strip() for s in args.disable_scorer.split(",") if s.strip()]
                            if args.disable_scorer else ()
                        ),
                        graph_balance=args.graph_balance,
                        normalize_terms=args.normalize_terms,
                    )
                    graph_result = grounder.ground(query_graph, top_k=args.top_k)
                    results = graph_result.get("results") or []
                    if results:
                        final_best_pred_id = int(results[0]["pred_id"])
                        if not args.disable_rerank:
                            rerank_pred_id, rerank_payload, rerank_raw = choose_reranked_pred_id(
                                client,
                                args.llm_model,
                                gt_rec.get("description") or "",
                                graph_result["query_graph"],
                                results[: max(int(args.rerank_top_k), 1)],
                            )
                            if rerank_pred_id is not None:
                                final_best_pred_id = rerank_pred_id
                except Exception as exc:  # noqa: BLE001
                    graph_error = str(exc)
            else:
                graph_error = parse_error

            dist = _compute_distance_for_prediction(gt_rec, pred_map, final_best_pred_id)

            # ---- grounding protocol: Coverage / SelAcc / error decomposition ----
            corr_thresh = args.corr_thresh
            gt_objs = scene_gt_objs.get(scene_id, {})
            covered = any(_corresponds(p, gt_rec, corr_thresh) for p in pred_map.values())
            sel_rec = pred_map.get(final_best_pred_id) if final_best_pred_id is not None else None
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
                "description": gt_rec.get("description"),
                "best_pred_id": final_best_pred_id,
                "distance": dist,
                "covered": covered,
                "sel_correct": correct,
                "error_type": error_type,
                "query_graph": _query_graph_to_dict(query_graph) if query_graph is not None else None,
                "graph_top_results": graph_result.get("results") if isinstance(graph_result, dict) else None,
                "graph_llm_context": graph_result.get("llm_context") if isinstance(graph_result, dict) else None,
                "resolved_classes": graph_result.get("resolved_classes") if isinstance(graph_result, dict) else None,
                "parse_error": parse_error,
                "graph_error": graph_error,
                "rerank_choice": rerank_payload,
                "parse_raw_response": parse_answer.strip(),
                "rerank_raw_response": rerank_raw,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if isinstance(graph_result, dict) and graph_result.get("results"):
                print(
                    json.dumps(
                        {
                            "scene_id": scene_id,
                            "best_pred_id": final_best_pred_id,
                            "top_score": graph_result["results"][0]["total_score"],
                            "best_direction": graph_result["results"][0].get("best_direction"),
                        },
                        ensure_ascii=False,
                    )
                )
            elif parse_payload is not None:
                print(json.dumps(parse_payload, ensure_ascii=False))
            else:
                print(parse_answer.strip())

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
        mean_dist, acc30, acc50, valid_total, total_count = metrics
        mean_part = f"{mean_dist:.4f}" if mean_dist is not None else "n/a"
        acc30_part = f"{acc30:.4f}" if acc30 is not None else "n/a"
        acc50_part = f"{acc50:.4f}" if acc50 is not None else "n/a"
        print(
            f"Eval: meanDist={mean_part}, Acc@0.3m={acc30_part}, Acc@0.5m={acc50_part} "
            f"| Coverage={n_covered / max(processed, 1):.4f}, "
            f"SelAcc@all={n_correct / max(processed, 1):.4f}, "
            f"SelAcc@covered={n_correct / max(n_covered, 1):.4f} "
            f"(valid={valid_total}/{total_count})"
        )
        metrics_rec = {
            "processed": processed,
            "skipped": skipped,
            "mean_dist": mean_dist,
            "acc03m": acc30,          # centroid distance <= 0.3 m
            "acc05m": acc50,          # centroid distance <= 0.5 m
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
