#!/usr/bin/env python3
"""
Graph-based spatial reasoning over 3D instance predictions.

This module builds a scene graph from object centers / sizes and scores
structured referring queries with explicit spatial relation operators.

Example query:
{
  "target_class": "kitchen cabinet",
  "viewer_direction": "auto",
  "relations": [
    {"type": "right_of", "ref_class": "stove", "weight": 1.0},
    {"type": "group_of", "count": 3, "weight": 0.7},
    {"type": "middle_in_group", "count": 3, "weight": 1.0},
    {"type": "upper", "weight": 0.5}
  ]
}
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


EPS = 1e-8
ALL_VIEWER_DIRECTIONS = ("+x", "-x", "+y", "-y")


RELATION_ALIASES = {
    "left": "left_of",
    "left_of": "left_of",
    "right": "right_of",
    "right_of": "right_of",
    "front": "front_of",
    "front_of": "front_of",
    "in_front_of": "front_of",
    "back": "behind",
    "back_of": "behind",
    "behind": "behind",
    "near": "near",
    "next_to": "near",
    "beside": "near",
    "adjacent": "near",
    "by": "near",
    "around": "near",
    "closest": "closest_to",
    "nearest": "closest_to",
    "closest_to": "closest_to",
    "nearest_to": "closest_to",
    "farthest": "farthest_from",
    "furthest": "farthest_from",
    "farthest_from": "farthest_from",
    "furthest_from": "farthest_from",
    "between": "between",
    "leftmost": "leftmost",
    "rightmost": "rightmost",
    "frontmost": "frontmost",
    "backmost": "backmost",
    "center_of_room": "center_of_room",
    "middle_of_room": "center_of_room",
    "middle": "center_of_room",
    "above": "above",
    "over": "above",
    "on_top_of": "on_top_of",
    "below": "below",
    "beneath": "below",
    "under": "under",
    "against_wall": "against_wall",
    "on_wall": "on_wall",
    "corner": "corner",
    "in_corner": "corner",
    "side_of_room_with": "same_side_as",
    "same_side_as": "same_side_as",
    "with": "with_object",
    "has": "with_object",
    "with_object": "with_object",
    "has_on_it": "has_on_it",
    "with_on_it": "has_on_it",
    "group_of": "group_of",
    "middle_in_group": "middle_in_group",
    "end_of_row": "end_of_row",
    "upper": "upper",
    "lower": "lower",
}


COUNT_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return " ".join(text.split())


def _extract_count(text: str, default: int = 3) -> int:
    for token, value in COUNT_WORDS.items():
        if token in text:
            return value
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return default


def _label_variants(value: Any) -> set[str]:
    label = _normalize_label(value)
    variants = {label}
    if not label:
        return variants
    if label.endswith("ies") and len(label) > 3:
        variants.add(label[:-3] + "y")
    if label.endswith("es") and len(label) > 2:
        variants.add(label[:-2])
    if label.endswith("s") and len(label) > 1:
        variants.add(label[:-1])
    return variants


def _labels_match(query_label: Any, object_label: Any) -> bool:
    q = _label_variants(query_label)
    o = _label_variants(object_label)
    return bool(q & o)


def _as_vec3(value: Any) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected vec3, got {value!r}")
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _safe_sigmoid(value: float, scale: float) -> float:
    scale = max(abs(scale), EPS)
    scaled = max(min(value / scale, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-scaled))


def _safe_exp_neg(value: float, scale: float) -> float:
    scale = max(abs(scale), EPS)
    return math.exp(-max(float(value), 0.0) / scale)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean_pairwise_distance(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    total = 0.0
    count = 0
    for a, b in itertools.combinations(points, 2):
        total += float(np.linalg.norm(a - b))
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
    gap = abs(float(arr[target_index]) - best_value)
    gap_component = _safe_exp_neg(gap, scale)
    return _clamp01(0.6 * rank_component + 0.4 * gap_component)


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


@dataclass(frozen=True)
class SceneObject:
    pred_id: int
    class_name: str
    center: np.ndarray
    size: np.ndarray
    box_min: np.ndarray
    box_max: np.ndarray
    color_rgb: Optional[np.ndarray]
    num_points: int
    scene_id: Optional[str]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SceneObject":
        center = _as_vec3(record.get("center"))
        size = _as_vec3(record.get("size"))
        box_min = record.get("box_min")
        box_max = record.get("box_max")
        if isinstance(box_min, (list, tuple)) and isinstance(box_max, (list, tuple)):
            bmin = _as_vec3(box_min)
            bmax = _as_vec3(box_max)
        else:
            half = size / 2.0
            bmin = center - half
            bmax = center + half
        color = record.get("color_rgb")
        color_rgb = np.asarray(color, dtype=np.float64) if isinstance(color, (list, tuple)) and len(color) == 3 else None
        return cls(
            pred_id=int(record["pred_id"]),
            class_name=str(record.get("class_name") or "object"),
            center=center,
            size=size,
            box_min=bmin,
            box_max=bmax,
            color_rgb=color_rgb,
            num_points=int(record.get("num_points") or 0),
            scene_id=record.get("scene_id"),
        )

    @property
    def norm_class_name(self) -> str:
        return _normalize_label(self.class_name)

    @property
    def volume(self) -> float:
        return float(np.prod(np.maximum(self.size, 0.0)))

    def to_summary(self) -> Dict[str, Any]:
        return {
            "pred_id": self.pred_id,
            "class_name": self.class_name,
            "center": [float(x) for x in self.center],
            "size": [float(x) for x in self.size],
            "num_points": self.num_points,
        }


class SpatialSceneGraph:
    def __init__(self, objects: Sequence[SceneObject]):
        if not objects:
            raise ValueError("SpatialSceneGraph requires at least one object")
        self.objects = list(objects)
        self.objects_by_id = {obj.pred_id: obj for obj in self.objects}
        self.graph = nx.DiGraph()
        self.room_min = np.min(np.stack([obj.box_min for obj in self.objects], axis=0), axis=0)
        self.room_max = np.max(np.stack([obj.box_max for obj in self.objects], axis=0), axis=0)
        self.room_extent = np.maximum(self.room_max - self.room_min, EPS)
        self.room_center = (self.room_min + self.room_max) / 2.0
        self.room_diag_xy = float(np.linalg.norm(self.room_extent[:2]))
        self.room_diag_3d = float(np.linalg.norm(self.room_extent))
        self.horizontal_scale = max(0.15 * self.room_diag_xy, 0.25)
        self.vertical_scale = max(0.20 * float(self.room_extent[2]), 0.15)
        self.midpoint_scale = max(0.18 * self.room_diag_xy, 0.30)
        self.wall_scale = max(0.12 * self.room_diag_xy, 0.20)
        self.corner_scale = max(0.14 * self.room_diag_xy, 0.25)
        self.group_scale = max(0.14 * self.room_diag_xy, 0.25)
        self.align_scale = max(0.10 * self.room_diag_xy, 0.20)
        self._build_graph()

    @classmethod
    def from_prediction_records(cls, records: Sequence[Mapping[str, Any]]) -> "SpatialSceneGraph":
        return cls([SceneObject.from_record(record) for record in records])

    def _build_graph(self) -> None:
        for obj in self.objects:
            self.graph.add_node(
                self._obj_node_id(obj.pred_id),
                node_type="object",
                pred_id=obj.pred_id,
                class_name=obj.class_name,
                center=obj.center,
                size=obj.size,
            )
        self._add_anchor_nodes()
        self._add_pair_edges()
        self._add_anchor_edges()

    def _add_anchor_nodes(self) -> None:
        anchors = {
            "room_center": self.room_center,
            "floor": np.array([self.room_center[0], self.room_center[1], self.room_min[2]], dtype=np.float64),
            "ceiling": np.array([self.room_center[0], self.room_center[1], self.room_max[2]], dtype=np.float64),
            "wall_x_min": np.array([self.room_min[0], self.room_center[1], self.room_center[2]], dtype=np.float64),
            "wall_x_max": np.array([self.room_max[0], self.room_center[1], self.room_center[2]], dtype=np.float64),
            "wall_y_min": np.array([self.room_center[0], self.room_min[1], self.room_center[2]], dtype=np.float64),
            "wall_y_max": np.array([self.room_center[0], self.room_max[1], self.room_center[2]], dtype=np.float64),
            "corner_xmin_ymin": np.array([self.room_min[0], self.room_min[1], self.room_min[2]], dtype=np.float64),
            "corner_xmin_ymax": np.array([self.room_min[0], self.room_max[1], self.room_min[2]], dtype=np.float64),
            "corner_xmax_ymin": np.array([self.room_max[0], self.room_min[1], self.room_min[2]], dtype=np.float64),
            "corner_xmax_ymax": np.array([self.room_max[0], self.room_max[1], self.room_min[2]], dtype=np.float64),
        }
        for name, position in anchors.items():
            self.graph.add_node(
                self._anchor_node_id(name),
                node_type="anchor",
                anchor_name=name,
                center=position,
            )

    def _add_pair_edges(self) -> None:
        for src in self.objects:
            for dst in self.objects:
                if src.pred_id == dst.pred_id:
                    continue
                delta = src.center - dst.center
                xy_iou = _xy_iou(src.box_min, src.box_max, dst.box_min, dst.box_max)
                gap_xy = float(np.linalg.norm(np.maximum(np.maximum(dst.box_min[:2] - src.box_max[:2], src.box_min[:2] - dst.box_max[:2]), 0.0)))
                z_gap = 0.0
                if src.box_min[2] > dst.box_max[2]:
                    z_gap = float(src.box_min[2] - dst.box_max[2])
                elif dst.box_min[2] > src.box_max[2]:
                    z_gap = float(dst.box_min[2] - src.box_max[2])
                self.graph.add_edge(
                    self._obj_node_id(src.pred_id),
                    self._obj_node_id(dst.pred_id),
                    edge_type="object_to_object",
                    dx=float(delta[0]),
                    dy=float(delta[1]),
                    dz=float(delta[2]),
                    distance_xy=float(np.linalg.norm(delta[:2])),
                    distance_3d=float(np.linalg.norm(delta)),
                    xy_iou=xy_iou,
                    gap_xy=gap_xy,
                    gap_z=z_gap,
                    same_class=src.norm_class_name == dst.norm_class_name,
                )

    def _add_anchor_edges(self) -> None:
        for obj in self.objects:
            wall_dist = self.wall_distances(obj)
            anchor_payloads = {
                "room_center": {
                    "distance_xy": float(np.linalg.norm(obj.center[:2] - self.room_center[:2])),
                    "distance_3d": float(np.linalg.norm(obj.center - self.room_center)),
                },
                "floor": {"distance": float(max(obj.box_min[2] - self.room_min[2], 0.0))},
                "ceiling": {"distance": float(max(self.room_max[2] - obj.box_max[2], 0.0))},
                "wall_x_min": {"distance": wall_dist["x_min"]},
                "wall_x_max": {"distance": wall_dist["x_max"]},
                "wall_y_min": {"distance": wall_dist["y_min"]},
                "wall_y_max": {"distance": wall_dist["y_max"]},
            }
            for anchor_name, attrs in anchor_payloads.items():
                self.graph.add_edge(
                    self._obj_node_id(obj.pred_id),
                    self._anchor_node_id(anchor_name),
                    edge_type="object_to_anchor",
                    **attrs,
                )

    @staticmethod
    def _obj_node_id(pred_id: int) -> str:
        return f"obj:{pred_id}"

    @staticmethod
    def _anchor_node_id(name: str) -> str:
        return f"anchor:{name}"

    def get_object(self, pred_id: int) -> SceneObject:
        return self.objects_by_id[int(pred_id)]

    def object_edge(self, src_id: int, dst_id: int) -> Dict[str, Any]:
        return dict(self.graph[self._obj_node_id(src_id)][self._obj_node_id(dst_id)])

    def resolve_objects(
        self,
        *,
        class_name: Optional[str] = None,
        pred_ids: Optional[Sequence[int]] = None,
    ) -> List[SceneObject]:
        if pred_ids is not None:
            resolved = [self.objects_by_id[int(pred_id)] for pred_id in pred_ids if int(pred_id) in self.objects_by_id]
            if class_name is None:
                return resolved
            return [obj for obj in resolved if _labels_match(class_name, obj.class_name)]
        if class_name is None:
            return list(self.objects)
        return [obj for obj in self.objects if _labels_match(class_name, obj.class_name)]

    def wall_distances(self, obj: SceneObject) -> Dict[str, float]:
        return {
            "x_min": float(max(obj.box_min[0] - self.room_min[0], 0.0)),
            "x_max": float(max(self.room_max[0] - obj.box_max[0], 0.0)),
            "y_min": float(max(obj.box_min[1] - self.room_min[1], 0.0)),
            "y_max": float(max(self.room_max[1] - obj.box_max[1], 0.0)),
        }

    def wall_affinity(self, obj: SceneObject) -> Dict[str, float]:
        return {name: _safe_exp_neg(distance, self.wall_scale) for name, distance in self.wall_distances(obj).items()}

    def nearest_corner_distance(self, obj: SceneObject) -> Tuple[float, str]:
        dists = self.wall_distances(obj)
        corner_scores = {
            "corner_xmin_ymin": math.hypot(dists["x_min"], dists["y_min"]),
            "corner_xmin_ymax": math.hypot(dists["x_min"], dists["y_max"]),
            "corner_xmax_ymin": math.hypot(dists["x_max"], dists["y_min"]),
            "corner_xmax_ymax": math.hypot(dists["x_max"], dists["y_max"]),
        }
        best_name = min(corner_scores, key=corner_scores.get)
        return float(corner_scores[best_name]), best_name

    def room_summary(self) -> Dict[str, Any]:
        return {
            "room_min": [float(x) for x in self.room_min],
            "room_max": [float(x) for x in self.room_max],
            "room_center": [float(x) for x in self.room_center],
            "num_objects": len(self.objects),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
        }


class SceneGraphQueryEngine:
    def __init__(self, scene_graph: SpatialSceneGraph):
        self.scene = scene_graph

    def score_query(self, query: Mapping[str, Any], top_k: int = 5) -> Dict[str, Any]:
        target_class = query.get("target_class")
        fallback_to_all = bool(query.get("fallback_to_all", True))
        candidate_ids = query.get("candidate_ids")
        if candidate_ids is not None:
            candidates = self.scene.resolve_objects(pred_ids=candidate_ids, class_name=target_class)
        else:
            candidates = self.scene.resolve_objects(class_name=target_class)
        if not candidates and fallback_to_all:
            candidates = self.scene.resolve_objects()
        if not candidates:
            return {
                "scene_id": candidates[0].scene_id if candidates else None,
                "query": dict(query),
                "room": self.scene.room_summary(),
                "results": [],
                "llm_context": "",
            }

        relations = [self._normalize_relation(rel) for rel in query.get("relations", [])]
        requested_direction = str(query.get("viewer_direction", "auto")).strip().lower()
        directions = ALL_VIEWER_DIRECTIONS if requested_direction in {"", "auto", "none", "null"} else (requested_direction,)

        results = []
        for candidate_index, candidate in enumerate(candidates):
            best_candidate_result = None
            for direction in directions:
                relation_results = []
                for relation in relations:
                    relation_result = self._score_relation(
                        candidate=candidate,
                        candidate_index=candidate_index,
                        target_candidates=candidates,
                        relation=relation,
                        viewer_direction=direction,
                    )
                    relation_results.append(relation_result)
                total_score = _geometric_mean(
                    [rel_result["score"] for rel_result in relation_results],
                    [rel_result["weight"] for rel_result in relation_results],
                ) if relation_results else 1.0
                candidate_result = {
                    "pred_id": candidate.pred_id,
                    "class_name": candidate.class_name,
                    "center": [float(x) for x in candidate.center],
                    "size": [float(x) for x in candidate.size],
                    "best_direction": direction if any(rel["uses_viewer_direction"] for rel in relation_results) else None,
                    "total_score": float(total_score),
                    "constraint_scores": relation_results,
                }
                if best_candidate_result is None or candidate_result["total_score"] > best_candidate_result["total_score"]:
                    best_candidate_result = candidate_result
            results.append(best_candidate_result)

        results.sort(key=lambda item: item["total_score"], reverse=True)
        top_results = results[: max(int(top_k), 1)]
        return {
            "scene_id": candidates[0].scene_id,
            "query": dict(query),
            "room": self.scene.room_summary(),
            "results": top_results,
            "llm_context": self.format_results_for_llm(top_results),
        }

    def format_results_for_llm(self, results: Sequence[Mapping[str, Any]]) -> str:
        lines: List[str] = []
        for result in results:
            parts = [
                f"pred_id={result['pred_id']}",
                f"class={result['class_name']}",
                f"total={result['total_score']:.3f}",
            ]
            if result.get("best_direction"):
                parts.append(f"dir={result['best_direction']}")
            for constraint in result.get("constraint_scores", []):
                parts.append(f"{constraint['type']}={constraint['score']:.3f}")
            lines.append(", ".join(parts))
        return "\n".join(lines)

    def _normalize_relation(self, relation: Mapping[str, Any]) -> Dict[str, Any]:
        if "type" not in relation:
            raise ValueError(f"Relation is missing 'type': {relation}")
        rel_type = _normalize_label(relation["type"]).replace(" ", "_")
        rel_type = RELATION_ALIASES.get(rel_type, rel_type)
        normalized = dict(relation)
        normalized["type"] = rel_type
        normalized["weight"] = float(relation.get("weight", 1.0))
        return normalized

    def _score_relation(
        self,
        *,
        candidate: SceneObject,
        candidate_index: int,
        target_candidates: Sequence[SceneObject],
        relation: Mapping[str, Any],
        viewer_direction: str,
    ) -> Dict[str, Any]:
        rel_type = str(relation["type"])
        weight = float(relation.get("weight", 1.0))
        if rel_type in {"left_of", "right_of", "front_of", "behind", "near", "closest_to", "farthest_from", "above", "below", "under", "on_top_of", "same_side_as", "with_object", "has_on_it"}:
            result = self._score_reference_relation(
                candidate=candidate,
                candidate_index=candidate_index,
                target_candidates=target_candidates,
                relation=relation,
                viewer_direction=viewer_direction,
            )
        elif rel_type == "between":
            result = self._score_between(candidate=candidate, relation=relation)
        elif rel_type in {"leftmost", "rightmost", "frontmost", "backmost", "center_of_room", "upper", "lower", "against_wall", "on_wall", "corner"}:
            result = self._score_unary_relation(
                candidate=candidate,
                candidate_index=candidate_index,
                target_candidates=target_candidates,
                relation=relation,
                viewer_direction=viewer_direction,
            )
        elif rel_type in {"group_of", "middle_in_group", "end_of_row"}:
            result = self._score_group_relation(candidate=candidate, relation=relation)
        else:
            raise ValueError(f"Unsupported relation type: {rel_type}")
        result["type"] = rel_type
        result["weight"] = weight
        return result

    def _score_reference_relation(
        self,
        *,
        candidate: SceneObject,
        candidate_index: int,
        target_candidates: Sequence[SceneObject],
        relation: Mapping[str, Any],
        viewer_direction: str,
    ) -> Dict[str, Any]:
        rel_type = str(relation["type"])
        refs = self._resolve_reference_objects(relation)
        refs = [ref for ref in refs if ref.pred_id != candidate.pred_id]
        if not refs:
            return {"score": 0.0, "uses_viewer_direction": rel_type in {"left_of", "right_of", "front_of", "behind"}}

        best_score = -1.0
        best_detail: Dict[str, Any] = {"ref_pred_ids": []}
        for ref in refs:
            if rel_type == "left_of":
                score = self._score_left_right(candidate, ref, viewer_direction, want="left")
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "right_of":
                score = self._score_left_right(candidate, ref, viewer_direction, want="right")
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "front_of":
                score = self._score_front_back(candidate, ref, viewer_direction, want="front")
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "behind":
                score = self._score_front_back(candidate, ref, viewer_direction, want="back")
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "near":
                score = self._score_near(candidate, ref)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "closest_to":
                values = [self._distance_3d(obj, ref) for obj in target_candidates]
                score = _soft_rank_score(values, candidate_index, lower_is_better=True, scale=self.scene.horizontal_scale)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "farthest_from":
                values = [self._distance_3d(obj, ref) for obj in target_candidates]
                score = _soft_rank_score(values, candidate_index, lower_is_better=False, scale=self.scene.horizontal_scale)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "above":
                score = self._score_above(candidate, ref, require_contact=False)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "below":
                score = self._score_below(candidate, ref, require_contact=False)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "under":
                score = self._score_below(candidate, ref, require_contact=True)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "on_top_of":
                score = self._score_above(candidate, ref, require_contact=True)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "same_side_as":
                score = self._score_same_side(candidate, ref)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "with_object":
                score = self._score_near(candidate, ref)
                detail = {"ref_pred_ids": [ref.pred_id]}
            elif rel_type == "has_on_it":
                score = self._score_above(ref, candidate, require_contact=True)
                detail = {"ref_pred_ids": [ref.pred_id]}
            else:
                raise ValueError(rel_type)
            if score > best_score:
                best_score = score
                best_detail = detail

        best_detail["score"] = float(best_score)
        best_detail["uses_viewer_direction"] = rel_type in {"left_of", "right_of", "front_of", "behind"}
        return best_detail

    def _score_between(self, *, candidate: SceneObject, relation: Mapping[str, Any]) -> Dict[str, Any]:
        refs_a = self._resolve_reference_objects(relation, prefix="ref_a")
        refs_b = self._resolve_reference_objects(relation, prefix="ref_b")
        best_score = 0.0
        best_detail: Dict[str, Any] = {"ref_pred_ids": []}
        for ref_a in refs_a:
            for ref_b in refs_b:
                if ref_a.pred_id == ref_b.pred_id:
                    continue
                if candidate.pred_id in {ref_a.pred_id, ref_b.pred_id}:
                    continue
                midpoint = (ref_a.center + ref_b.center) / 2.0
                midpoint_score = _safe_exp_neg(float(np.linalg.norm(candidate.center - midpoint)), self.scene.midpoint_scale)
                balance_score = _safe_exp_neg(
                    abs(self._distance_3d(candidate, ref_a) - self._distance_3d(candidate, ref_b)),
                    self.scene.midpoint_scale,
                )
                line_score = _safe_exp_neg(
                    _point_segment_distance_xy(candidate.center, ref_a.center, ref_b.center),
                    self.scene.align_scale,
                )
                score = _clamp01(midpoint_score * balance_score * line_score)
                if score > best_score:
                    best_score = score
                    best_detail = {"ref_pred_ids": [ref_a.pred_id, ref_b.pred_id]}
        best_detail["score"] = float(best_score)
        best_detail["uses_viewer_direction"] = False
        return best_detail

    def _score_unary_relation(
        self,
        *,
        candidate: SceneObject,
        candidate_index: int,
        target_candidates: Sequence[SceneObject],
        relation: Mapping[str, Any],
        viewer_direction: str,
    ) -> Dict[str, Any]:
        rel_type = str(relation["type"])
        if rel_type == "center_of_room":
            score = _safe_exp_neg(float(np.linalg.norm(candidate.center - self.scene.room_center)), self.scene.midpoint_scale)
        elif rel_type == "against_wall":
            score = max(self.scene.wall_affinity(candidate).values())
        elif rel_type == "on_wall":
            against = max(self.scene.wall_affinity(candidate).values())
            clearance = float(max(candidate.box_min[2] - self.scene.room_min[2], 0.0))
            elevated = 1.0 - _safe_exp_neg(clearance, self.scene.vertical_scale)
            score = _clamp01(against * elevated)
        elif rel_type == "corner":
            dist, corner_name = self.scene.nearest_corner_distance(candidate)
            score = _safe_exp_neg(dist, self.scene.corner_scale)
            return {
                "score": float(score),
                "uses_viewer_direction": False,
                "corner": corner_name,
            }
        elif rel_type == "upper":
            values = [obj.center[2] for obj in target_candidates]
            score = _soft_rank_score(values, candidate_index, lower_is_better=False, scale=self.scene.vertical_scale)
        elif rel_type == "lower":
            values = [obj.center[2] for obj in target_candidates]
            score = _soft_rank_score(values, candidate_index, lower_is_better=True, scale=self.scene.vertical_scale)
        elif rel_type in {"leftmost", "rightmost", "frontmost", "backmost"}:
            front_axis, left_axis = _axis_from_viewer_direction(viewer_direction)
            centers = np.stack([obj.center for obj in target_candidates], axis=0)
            if rel_type == "leftmost":
                values = [float(np.dot(center, left_axis)) for center in centers]
                score = _soft_rank_score(values, candidate_index, lower_is_better=False, scale=self.scene.align_scale)
            elif rel_type == "rightmost":
                values = [float(np.dot(center, left_axis)) for center in centers]
                score = _soft_rank_score(values, candidate_index, lower_is_better=True, scale=self.scene.align_scale)
            elif rel_type == "frontmost":
                values = [float(np.dot(center, front_axis)) for center in centers]
                score = _soft_rank_score(values, candidate_index, lower_is_better=False, scale=self.scene.align_scale)
            else:
                values = [float(np.dot(center, front_axis)) for center in centers]
                score = _soft_rank_score(values, candidate_index, lower_is_better=True, scale=self.scene.align_scale)
        else:
            raise ValueError(rel_type)
        return {"score": float(score), "uses_viewer_direction": rel_type in {"leftmost", "rightmost", "frontmost", "backmost"}}

    def _score_group_relation(self, *, candidate: SceneObject, relation: Mapping[str, Any]) -> Dict[str, Any]:
        rel_type = str(relation["type"])
        count = int(relation.get("count", 3))
        same_class = self.scene.resolve_objects(class_name=relation.get("group_class", candidate.class_name))
        local_group = self._local_same_class_group(candidate, same_class, count=count)
        if rel_type == "group_of":
            score, group_ids = self._best_group_membership(candidate, local_group, count=count)
        elif rel_type == "middle_in_group":
            score, group_ids = self._best_middle_in_group(candidate, local_group, count=count)
        elif rel_type == "end_of_row":
            score, group_ids = self._best_end_of_row(candidate, local_group, count=max(count, 3))
        else:
            raise ValueError(rel_type)
        return {"score": float(score), "uses_viewer_direction": False, "group_pred_ids": group_ids}

    def _resolve_reference_objects(self, relation: Mapping[str, Any], prefix: str = "ref") -> List[SceneObject]:
        pred_id_key = f"{prefix}_pred_id"
        pred_ids_key = f"{prefix}_pred_ids"
        class_key = f"{prefix}_class"
        classes_key = f"{prefix}_classes"

        if pred_id_key in relation:
            return self.scene.resolve_objects(pred_ids=[int(relation[pred_id_key])])
        if pred_ids_key in relation:
            return self.scene.resolve_objects(pred_ids=[int(value) for value in relation[pred_ids_key]])
        if class_key in relation:
            return self.scene.resolve_objects(class_name=str(relation[class_key]))
        if classes_key in relation:
            refs: List[SceneObject] = []
            for class_name in relation[classes_key]:
                refs.extend(self.scene.resolve_objects(class_name=str(class_name)))
            return refs
        return []

    def _distance_3d(self, a: SceneObject, b: SceneObject) -> float:
        return float(np.linalg.norm(a.center - b.center))

    def _score_left_right(self, candidate: SceneObject, ref: SceneObject, viewer_direction: str, want: str) -> float:
        _, left_axis = _axis_from_viewer_direction(viewer_direction)
        front_axis, _ = _axis_from_viewer_direction(viewer_direction)
        delta = candidate.center - ref.center
        left_proj = float(np.dot(delta, left_axis))
        front_proj = float(np.dot(delta, front_axis))
        sign = 1.0 if want == "left" else -1.0
        dir_score = _safe_sigmoid(sign * left_proj, self.scene.horizontal_scale)
        orth_penalty = _safe_exp_neg(abs(front_proj), self.scene.horizontal_scale * 1.5)
        return _clamp01(dir_score * orth_penalty)

    def _score_front_back(self, candidate: SceneObject, ref: SceneObject, viewer_direction: str, want: str) -> float:
        front_axis, left_axis = _axis_from_viewer_direction(viewer_direction)
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

    def _local_same_class_group(self, candidate: SceneObject, same_class: Sequence[SceneObject], count: int) -> List[SceneObject]:
        neighbors = [obj for obj in same_class if obj.pred_id != candidate.pred_id]
        neighbors.sort(key=lambda obj: float(np.linalg.norm(obj.center - candidate.center)))
        limit = max(count + 3, 6)
        return [candidate] + neighbors[: max(limit - 1, 0)]

    def _best_group_membership(self, candidate: SceneObject, local_group: Sequence[SceneObject], count: int) -> Tuple[float, List[int]]:
        if len(local_group) < count or count <= 0:
            return 0.0, []
        candidate_id = candidate.pred_id
        best_score = 0.0
        best_group: List[int] = []
        local_by_id = {obj.pred_id: obj for obj in local_group}
        other_ids = [obj.pred_id for obj in local_group if obj.pred_id != candidate_id]
        for subset in itertools.combinations(other_ids, count - 1):
            group_ids = [candidate_id, *subset]
            points = np.stack([local_by_id[pred_id].center for pred_id in group_ids], axis=0)
            compactness = _safe_exp_neg(_mean_pairwise_distance(points), self.scene.group_scale)
            if compactness > best_score:
                best_score = compactness
                best_group = group_ids
        return best_score, best_group

    def _best_middle_in_group(self, candidate: SceneObject, local_group: Sequence[SceneObject], count: int) -> Tuple[float, List[int]]:
        if len(local_group) < count or count <= 1:
            return 0.0, []
        candidate_id = candidate.pred_id
        best_score = 0.0
        best_group: List[int] = []
        local_by_id = {obj.pred_id: obj for obj in local_group}
        other_ids = [obj.pred_id for obj in local_group if obj.pred_id != candidate_id]
        for subset in itertools.combinations(other_ids, count - 1):
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

    def _best_end_of_row(self, candidate: SceneObject, local_group: Sequence[SceneObject], count: int) -> Tuple[float, List[int]]:
        if len(local_group) < count or count < 3:
            return 0.0, []
        candidate_id = candidate.pred_id
        best_score = 0.0
        best_group: List[int] = []
        local_by_id = {obj.pred_id: obj for obj in local_group}
        other_ids = [obj.pred_id for obj in local_group if obj.pred_id != candidate_id]
        for subset in itertools.combinations(other_ids, count - 1):
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


def compile_query_from_parsed_chain(
    parsed_chain: Sequence[Mapping[str, Any]],
    *,
    viewer_direction: Optional[str] = None,
    fallback_to_all: bool = True,
) -> Dict[str, Any]:
    refs = [node for node in parsed_chain if _normalize_label(node.get("role")) == "ref"]
    target = next((node for node in parsed_chain if _normalize_label(node.get("role")) == "target"), None)
    if target is None:
        raise ValueError("parsed_chain must contain one target node")

    query: Dict[str, Any] = {
        "target_class": target.get("class"),
        "viewer_direction": viewer_direction or "auto",
        "fallback_to_all": fallback_to_all,
        "relations": [],
    }
    for constraint in target.get("constraints", []) or []:
        compiled = _compile_constraint_text(str(constraint), refs)
        if isinstance(compiled, list):
            query["relations"].extend(compiled)
        elif compiled is not None:
            query["relations"].append(compiled)
    return query


def _compile_constraint_text(constraint_text: str, refs: Sequence[Mapping[str, Any]]) -> Optional[Any]:
    text = _normalize_label(constraint_text)
    if not text:
        return None

    if text in {"upper", "top"}:
        return {"type": "upper"}
    if text in {"lower", "bottom"}:
        return {"type": "lower"}
    if text in {"leftmost", "rightmost", "frontmost", "backmost"}:
        return {"type": text}
    if "center of room" in text or "middle of room" in text:
        return {"type": "center_of_room"}
    if text in {"corner", "in corner"}:
        return {"type": "corner"}
    if "against wall" in text:
        return {"type": "against_wall"}
    if "on the wall" in text or text == "wall mounted":
        return {"type": "on_wall"}
    if "middle of group of" in text:
        count = _extract_count(text, default=3)
        return [{"type": "group_of", "count": count}, {"type": "middle_in_group", "count": count}]
    if text.startswith("group of"):
        return {"type": "group_of", "count": _extract_count(text, default=3)}
    if "end of row" in text or "end of line" in text:
        return {"type": "end_of_row", "count": _extract_count(text, default=3)}
    if (text.startswith("with ") or text.startswith("has ")) and text.endswith(" on it"):
        ref_class = _resolve_reference_hint(text.rsplit(" on it", 1)[0].split(" ", 1)[1].strip(), refs)
        return {"type": "has_on_it", "ref_class": ref_class} if ref_class is not None else {"type": "has_on_it"}

    relation_specs = [
        ("with", "with_object"),
        ("has", "with_object"),
        ("left of", "left_of"),
        ("right of", "right_of"),
        ("in front of", "front_of"),
        ("front of", "front_of"),
        ("behind", "behind"),
        ("next to", "near"),
        ("near", "near"),
        ("beside", "near"),
        ("adjacent to", "near"),
        ("closest to", "closest_to"),
        ("nearest to", "closest_to"),
        ("farthest from", "farthest_from"),
        ("furthest from", "farthest_from"),
        ("on top of", "on_top_of"),
        ("above", "above"),
        ("under", "under"),
        ("below", "below"),
        ("beneath", "below"),
        ("side of room with", "same_side_as"),
    ]
    for phrase, rel_type in relation_specs:
        if text.startswith(phrase):
            ref_class = _resolve_reference_hint(text[len(phrase):].strip(), refs)
            if ref_class is None and rel_type != "center_of_room":
                return {"type": rel_type}
            return {"type": rel_type, "ref_class": ref_class}

    if text.startswith("between"):
        a_class, b_class = _resolve_between_reference_hints(text, refs)
        if a_class and b_class:
            return {"type": "between", "ref_a_class": a_class, "ref_b_class": b_class}
        return {"type": "between"}

    return {"type": text.replace(" ", "_")}


def _resolve_reference_hint(text: str, refs: Sequence[Mapping[str, Any]]) -> Optional[str]:
    clean = text.strip()
    if not clean:
        return refs[0].get("class") if refs else None

    ref_match = re.search(r"\bref(\d+)?\b", clean)
    if ref_match and refs:
        ref_index = int(ref_match.group(1) or "1") - 1
        ref_index = max(0, min(ref_index, len(refs) - 1))
        return refs[ref_index].get("class")

    if clean.startswith("the "):
        clean = clean[4:]
    for ref in refs:
        ref_class = str(ref.get("class") or "").strip()
        if ref_class and ref_class in clean:
            return ref_class
    return clean or None


def _resolve_between_reference_hints(text: str, refs: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    ref_matches = list(re.finditer(r"\bref(\d+)?\b", text))
    if len(ref_matches) >= 2 and refs:
        resolved: List[Optional[str]] = []
        for match in ref_matches[:2]:
            ref_index = int(match.group(1) or "1") - 1
            ref_index = max(0, min(ref_index, len(refs) - 1))
            resolved.append(refs[ref_index].get("class"))
        return resolved[0], resolved[1]

    parts = text.split(" and ", 1)
    if len(parts) == 2:
        return _resolve_reference_hint(parts[0].replace("between", "").strip(), refs), _resolve_reference_hint(parts[1].strip(), refs)

    if len(refs) >= 2:
        return refs[0].get("class"), refs[1].get("class")
    return None, None


def load_query(args: argparse.Namespace) -> Dict[str, Any]:
    if args.query_json:
        return json.loads(args.query_json)
    if args.query_file:
        return json.loads(Path(args.query_file).read_text(encoding="utf-8"))
    raise ValueError("Either --query-json or --query-file must be provided")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score structured spatial queries over pred_bboxes.json")
    parser.add_argument("--preds", type=Path, required=True, help="Path to pred_bboxes.json")
    parser.add_argument("--query-json", default=None, help="Inline query JSON string")
    parser.add_argument("--query-file", type=Path, default=None, help="Path to query JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    records = json.loads(args.preds.read_text(encoding="utf-8"))
    query = load_query(args)
    scene_graph = SpatialSceneGraph.from_prediction_records(records)
    engine = SceneGraphQueryEngine(scene_graph)
    result = engine.score_query(query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
