#!/usr/bin/env python3
"""
Edge query-graph grounding for a single text description.

Pipeline:
1. Parse the description into a compact query graph with a local OpenAI-compatible LLM.
2. Load `objects.csv` and convert each row into a scene-graph prediction record.
3. Use a local TensorRT CLIP text encoder to align LLM class names with CSV labels.
4. Call `spatial_scene_graph.py` to score candidate objects.
5. Optionally ask the LLM to rerank a short list.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/usr/lib/python3.10/dist-packages/tensorrt")

import argparse
import ast
import csv
import gzip
import html
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except ImportError:  # pragma: no cover
    cuda = None

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover
    trt = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from spatial_scene_graph import SceneGraphQueryEngine, SpatialSceneGraph


DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_LLM_API_KEY = "EMPTY"
DEFAULT_LLM_MODEL = "qwen3-1.7b"
DEFAULT_VIEWER_DIRECTION = "+x"
DEFAULT_BPE_PATH = Path("/home/nvidia/fjr/LLM_models/engines/clip-vit-base-patch16/bpe_simple_vocab_16e6.txt.gz")
DEFAULT_OBJECTS_PATH = Path("/home/nvidia/.hydra/orbbec/backend/objects.csv")
DEFAULT_CLIP_PATH = Path("/home/nvidia/fjr/LLM_models/engines/clip-vit-base-patch16/clip_text_encoder.engine")

EPS = 1e-8
CONTEXT_LENGTH = 77
TEXT_PROMPT_TEMPLATES = (
    "{}",
    "an indoor object called {}",
    "a 3d indoor object called {}",
)
ALL_VIEWER_DIRECTIONS = ("+x", "-x", "+y", "-y")

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
    "with": "with_object",
    "with_object": "with_object",
    "has_on_it": "has_on_it",
    "between": "between",
}

ATTR_ALIASES = {
    "upper": "upper",
    "higher": "upper",
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
    "smallest": "smallest",
    "tallest": "tallest",
    "shortest": "shortest",
    "open": "open",
    "closed": "closed",
    "empty": "empty",
}

COLOR_ATTRS = {
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "brown",
    "pink",
    "purple",
    "silver",
}

TARGET_SCENE_ATTRS = {
    "upper",
    "lower",
    "leftmost",
    "rightmost",
    "frontmost",
    "backmost",
    "corner",
    "center_of_room",
    "against_wall",
    "on_wall",
}

POST_RERANK_ATTRS = {"largest", "smallest", "tallest", "shortest"}
FILTERABLE_REF_ATTRS = TARGET_SCENE_ATTRS | POST_RERANK_ATTRS
IGNORED_ATTRS = {"open", "closed", "empty"} | COLOR_ATTRS


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return " ".join(text.split())


def _normalize_relation_name(value: Any) -> str:
    text = _normalize_label(value).replace(" ", "_")
    return RELATION_ALIASES.get(text, text)


def _normalize_attribute_name(value: Any) -> str:
    text = _normalize_label(value).replace(" ", "_")
    return ATTR_ALIASES.get(text, text)


def _normalize_viewer_direction(value: Any) -> Optional[str]:
    text = _normalize_label(value)
    if text in {"", "none", "null"}:
        return None
    return text if text in ALL_VIEWER_DIRECTIONS else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_exp_neg(value: float, scale: float) -> float:
    scale = max(abs(scale), EPS)
    return math.exp(-max(float(value), 0.0) / scale)


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


def _geometric_mean(scores: Sequence[float]) -> float:
    if not scores:
        return 1.0
    acc = 0.0
    for score in scores:
        acc += math.log(max(float(score), 1e-6))
    return math.exp(acc / len(scores))


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
    return bool(_label_variants(query_label) & _label_variants(object_label))


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _trt_dtype_to_np(dtype: Any) -> np.dtype:
    if trt is None:
        raise RuntimeError("TensorRT is not available")
    mapping = {
        getattr(trt.DataType, "FLOAT", None): np.float32,
        getattr(trt.DataType, "HALF", None): np.float16,
        getattr(trt.DataType, "INT8", None): np.int8,
        getattr(trt.DataType, "INT32", None): np.int32,
        getattr(trt.DataType, "BOOL", None): np.bool_,
        getattr(trt.DataType, "UINT8", None): np.uint8,
        getattr(trt.DataType, "BF16", None): np.float16,
        getattr(trt.DataType, "FP8", None): np.float16,
    }
    np_dtype = mapping.get(dtype)
    if np_dtype is None:
        raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")
    return np.dtype(np_dtype)


def _parse_vector_text(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            try:
                return [float(x) for x in parsed]
            except (TypeError, ValueError):
                return None
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 3:
            try:
                return [float(x) for x in parts]
            except ValueError:
                return None
    return None


def _vector_from_xyz(row: Mapping[str, Any], prefix: str) -> Optional[List[float]]:
    vals = [row.get(f"{prefix}_{axis}") for axis in ("x", "y", "z")]
    if any(v in {None, ""} for v in vals):
        return None
    try:
        return [float(v) for v in vals]
    except (TypeError, ValueError):
        return None


def _bbox_from_center_size(center: Sequence[float], size: Sequence[float]) -> Tuple[List[float], List[float]]:
    return (
        [float(c - s / 2.0) for c, s in zip(center, size)],
        [float(c + s / 2.0) for c, s in zip(center, size)],
    )


def _parse_csv_row_to_record(row: Mapping[str, Any], row_index: int, scene_id_override: Optional[str]) -> Optional[Dict[str, Any]]:
    class_name = (
        row.get("label")
        or row.get("class_name")
        or row.get("class")
        or row.get("name")
        or ""
    )
    class_name = _normalize_label(class_name)
    if not class_name:
        return None

    center = (
        _vector_from_xyz(row, "bbox_center")
        or _vector_from_xyz(row, "center")
        or _parse_vector_text(row.get("bbox_center"))
        or _parse_vector_text(row.get("center"))
    )
    size = (
        _vector_from_xyz(row, "bbox_dim")
        or _vector_from_xyz(row, "size")
        or _parse_vector_text(row.get("bbox_dim"))
        or _parse_vector_text(row.get("size"))
    )
    box_min = _vector_from_xyz(row, "box_min") or _parse_vector_text(row.get("box_min"))
    box_max = _vector_from_xyz(row, "box_max") or _parse_vector_text(row.get("box_max"))

    if center is None and box_min is not None and box_max is not None:
        center = [float((box_min[i] + box_max[i]) / 2.0) for i in range(3)]
    if size is None and box_min is not None and box_max is not None:
        size = [float(box_max[i] - box_min[i]) for i in range(3)]
    if center is None or size is None:
        return None
    if box_min is None or box_max is None:
        box_min, box_max = _bbox_from_center_size(center, size)

    scene_id = (
        scene_id_override
        or str(row.get("scene_id") or "").strip()
        or str(row.get("id") or "").strip()
        or None
    )
    num_points = int(float(row.get("num_points") or row.get("points") or 0))
    return {
        "pred_id": row_index,
        "class_name": class_name,
        "center": center,
        "size": size,
        "box_min": box_min,
        "box_max": box_max,
        "num_points": num_points,
        "scene_id": scene_id,
        "raw_row": dict(row),
    }


def load_prediction_records_from_csv(path: Path, scene_id_override: Optional[str] = None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        records: List[Dict[str, Any]] = []
        for row_index, row in enumerate(reader, 1):
            record = _parse_csv_row_to_record(row, row_index=row_index, scene_id_override=scene_id_override)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError(f"No valid prediction rows found in {path}")
    return records


def build_class_inventory(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        class_name = _normalize_label(record.get("class_name"))
        if class_name:
            counts[class_name] += 1
    return [{"class_name": name, "count": count} for name, count in sorted(counts.items())]


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


def make_client(base_url: str, api_key: str) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("The `openai` package is required to use the local LLM endpoint.")
    return OpenAI(base_url=base_url, api_key=api_key)


@dataclass(frozen=True)
class QueryNode:
    node_id: str
    role: str
    class_name: Optional[str]
    attributes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryRelation:
    rel_type: str
    source: str
    targets: Tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class QueryGraph:
    viewer_direction: Optional[str]
    parser_confidence: Optional[float]
    reason: Optional[str]
    nodes: Tuple[QueryNode, ...]
    relations: Tuple[QueryRelation, ...]

    def target_node(self) -> QueryNode:
        targets = [node for node in self.nodes if node.role == "target"]
        if len(targets) != 1:
            raise ValueError("Query graph must contain exactly one target node")
        return targets[0]

    def node_map(self) -> Dict[str, QueryNode]:
        return {node.node_id: node for node in self.nodes}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QueryGraph":
        raw_nodes = payload.get("nodes") or []
        raw_relations = payload.get("relations") or []
        if not raw_nodes:
            raise ValueError("Parser payload is missing `nodes`")

        nodes: List[QueryNode] = []
        target_seen = False
        ref_index = 0
        for raw_node in raw_nodes:
            role = _normalize_label(raw_node.get("role"))
            if role not in {"target", "ref"}:
                continue
            if role == "target":
                if target_seen:
                    raise ValueError("Parser payload contains more than one target node")
                target_seen = True
                node_id = "t"
            else:
                ref_index += 1
                node_id = str(raw_node.get("id") or f"r{ref_index}")
            attrs = []
            for attr in raw_node.get("attributes") or raw_node.get("attrs") or []:
                norm_attr = _normalize_attribute_name(attr)
                if norm_attr:
                    attrs.append(norm_attr)
            nodes.append(
                QueryNode(
                    node_id=str(raw_node.get("id") or node_id),
                    role=role,
                    class_name=_normalize_label(raw_node.get("class")),
                    attributes=tuple(attrs),
                )
            )
        if not target_seen:
            raise ValueError("Parser payload is missing target node `t`")

        relations: List[QueryRelation] = []
        for raw_rel in raw_relations:
            rel_type = _normalize_relation_name(raw_rel.get("type"))
            source = str(raw_rel.get("source") or "")
            targets: List[str] = []
            if raw_rel.get("target"):
                targets.append(str(raw_rel["target"]))
            for value in raw_rel.get("targets") or []:
                targets.append(str(value))
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
            viewer_direction=_normalize_viewer_direction(payload.get("viewer_direction")),
            parser_confidence=_coerce_float(payload.get("parser_confidence")),
            reason=str(payload.get("reason") or "").strip() or None,
            nodes=tuple(nodes),
            relations=tuple(relations),
        )


def query_graph_to_dict(query_graph: QueryGraph) -> Dict[str, Any]:
    return {
        "viewer_direction": query_graph.viewer_direction,
        "parser_confidence": query_graph.parser_confidence,
        "reason": query_graph.reason,
        "nodes": [
            {
                "id": node.node_id,
                "role": node.role,
                "class": node.class_name,
                "attributes": list(node.attributes),
            }
            for node in query_graph.nodes
        ],
        "relations": [
            {
                "type": rel.rel_type,
                "source": rel.source,
                "targets": list(rel.targets),
                "weight": rel.weight,
            }
            for rel in query_graph.relations
        ],
    }


@lru_cache(maxsize=1)
def bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class SimpleTokenizer:
    def __init__(self, bpe_path: Path) -> None:
        bpe_merges = gzip.open(bpe_path, "rt", encoding="utf-8").read().splitlines()
        bpe_merges = bpe_merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in bpe_merges if merge]

        vocab = list(bytes_to_unicode().values())
        vocab += [v + "</w>" for v in vocab]
        vocab += ["".join(merge) for merge in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]
        self.pat = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|\d+|[^\sA-Za-z\d]+",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text: str) -> List[int]:
        text = " ".join(html.unescape(html.unescape(text or "")).strip().split()).lower()
        bpe_tokens: List[int] = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def tokenize(self, texts: Sequence[str], context_length: int = CONTEXT_LENGTH) -> np.ndarray:
        tokens = np.zeros((len(texts), context_length), dtype=np.int32)
        for row_index, text in enumerate(texts):
            encoded = [self.sot_token, *self.encode(text), self.eot_token]
            encoded = encoded[:context_length]
            tokens[row_index, : len(encoded)] = np.asarray(encoded, dtype=np.int32)
        return tokens


class TRTCLIPTextMatcher:
    def __init__(
        self,
        engine_path: Path,
        bpe_path: Path,
        *,
        output_name: Optional[str] = None,
        pooling: str = "auto",
    ) -> None:
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT + PyCUDA are required for TRT CLIP matching.")

        logger = trt.Logger(trt.Logger.WARNING)
        try:
            trt.init_libnvinfer_plugins(logger, "")
        except Exception:  # noqa: BLE001
            pass
        with engine_path.open("rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.tokenizer = SimpleTokenizer(bpe_path)
        self.pooling = pooling

        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("This script expects TensorRT 10.x tensor I/O API.")

        tensor_mode = trt.TensorIOMode
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == tensor_mode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == tensor_mode.OUTPUT
        ]
        if not self.input_names:
            raise RuntimeError("TensorRT engine has no input bindings.")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors.")

        self.input_ids_name = next((name for name in self.input_names if "input_ids" in name), self.input_names[0])
        self.attention_mask_name = next(
            (name for name in self.input_names if "attention_mask" in name or name.endswith("mask")),
            None,
        )
        if output_name is not None:
            if output_name not in self.output_names:
                raise RuntimeError(f"Failed to find TensorRT output tensor: {output_name}")
            self.output_name = output_name
        else:
            self.output_name = self.output_names[0]
        self.input_dtype = _trt_dtype_to_np(self.engine.get_tensor_dtype(self.input_ids_name))
        self.eot_token = self.tokenizer.eot_token
        self._cache: Dict[str, np.ndarray] = {}

    def _infer(self, token_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        token_ids = np.ascontiguousarray(token_ids.astype(self.input_dtype, copy=False))
        if attention_mask is not None:
            attention_mask = np.ascontiguousarray(attention_mask.astype(self.input_dtype, copy=False))
        self.context.set_input_shape(self.input_ids_name, tuple(token_ids.shape))
        if self.attention_mask_name is not None and attention_mask is not None:
            self.context.set_input_shape(self.attention_mask_name, tuple(attention_mask.shape))

        device_buffers: Dict[str, Any] = {}
        host_outputs: Dict[str, np.ndarray] = {}

        input_ids_buffer = cuda.mem_alloc(token_ids.nbytes)
        device_buffers[self.input_ids_name] = input_ids_buffer
        self.context.set_tensor_address(self.input_ids_name, int(input_ids_buffer))

        if self.attention_mask_name is not None and attention_mask is not None:
            attention_buffer = cuda.mem_alloc(attention_mask.nbytes)
            device_buffers[self.attention_mask_name] = attention_buffer
            self.context.set_tensor_address(self.attention_mask_name, int(attention_buffer))

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_np(self.engine.get_tensor_dtype(name))
            host_arr = np.empty(shape, dtype=dtype)
            dev_arr = cuda.mem_alloc(host_arr.nbytes)
            device_buffers[name] = dev_arr
            host_outputs[name] = host_arr
            self.context.set_tensor_address(name, int(dev_arr))

        cuda.memcpy_htod_async(input_ids_buffer, token_ids, self.stream)
        if self.attention_mask_name is not None and attention_mask is not None:
            cuda.memcpy_htod_async(device_buffers[self.attention_mask_name], attention_mask, self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for name, host_arr in host_outputs.items():
            cuda.memcpy_dtoh_async(host_arr, device_buffers[name], self.stream)
        self.stream.synchronize()
        return host_outputs[self.output_name]

    def _pool_hidden(self, hidden: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden)
        if hidden.ndim == 2:
            return hidden.astype(np.float32, copy=False)
        if hidden.ndim != 3:
            raise RuntimeError(f"Unexpected TensorRT output shape: {tuple(hidden.shape)}")

        method = self.pooling
        if method == "auto":
            method = "eot" if hidden.shape[1] == token_ids.shape[1] else "mean"

        if method == "first":
            return hidden[:, 0, :].astype(np.float32, copy=False)
        if method == "mean":
            mask = (token_ids != 0).astype(np.float32)
            denom = np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
            return ((hidden * mask[:, :, None]).sum(axis=1) / denom).astype(np.float32, copy=False)

        pooled = np.empty((hidden.shape[0], hidden.shape[2]), dtype=np.float32)
        for row_index in range(hidden.shape[0]):
            eot_positions = np.where(token_ids[row_index] == self.eot_token)[0]
            token_index = int(eot_positions[-1]) if len(eot_positions) else int(max(np.count_nonzero(token_ids[row_index]) - 1, 0))
            pooled[row_index] = hidden[row_index, token_index]
        return pooled

    def _encode_prompts(self, prompts: Sequence[str]) -> np.ndarray:
        token_ids = self.tokenizer.tokenize(prompts)
        attention_mask = (token_ids != 0).astype(np.int32, copy=False)
        hidden = self._infer(token_ids, attention_mask)
        feats = self._pool_hidden(hidden, token_ids)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, EPS, None)
        return feats.astype(np.float32, copy=False)

    def encode_label(self, label: str) -> np.ndarray:
        label = _normalize_label(label)
        if not label:
            raise ValueError("Label for CLIP encoding must be non-empty")
        if label not in self._cache:
            prompts = [template.format(label) for template in TEXT_PROMPT_TEMPLATES]
            feats = self._encode_prompts(prompts)
            feat = feats.mean(axis=0)
            feat = feat / max(float(np.linalg.norm(feat)), EPS)
            self._cache[label] = feat.astype(np.float32, copy=False)
        return self._cache[label]

    def top_matches(self, query_text: str, candidate_texts: Sequence[str], top_k: int = 3) -> List[Tuple[str, float]]:
        query_text = _normalize_label(query_text)
        labels = []
        seen = set()
        for label in candidate_texts:
            norm = _normalize_label(label)
            if norm and norm not in seen:
                seen.add(norm)
                labels.append(norm)
        if not query_text or not labels:
            return []
        query_feat = self.encode_label(query_text)
        label_feats = np.stack([self.encode_label(label) for label in labels], axis=0)
        sims = label_feats @ query_feat
        order = np.argsort(-sims)
        return [(labels[int(idx)], float(sims[int(idx)])) for idx in order[: max(int(top_k), 1)]]


@dataclass(frozen=True)
class ResolvedClass:
    class_name: str
    similarity: float


@dataclass(frozen=True)
class EdgeGroundingConfig:
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = DEFAULT_LLM_API_KEY
    llm_model: str = DEFAULT_LLM_MODEL
    clip_engine: Path = DEFAULT_CLIP_PATH
    bpe_path: Path = DEFAULT_BPE_PATH
    default_pred_csv: Path = DEFAULT_OBJECTS_PATH
    default_scene_id: Optional[str] = None
    engine_output_name: Optional[str] = None
    engine_output_pooling: str = "auto"
    viewer_direction: str = DEFAULT_VIEWER_DIRECTION
    respect_llm_viewer_direction: bool = False
    top_k: int = 5
    rerank_top_k: int = 4
    disable_rerank: bool = False
    class_top_k: int = 3
    min_clip_similarity: float = 0.18
    class_margin: float = 0.08
    llm_extra_body: Optional[Mapping[str, Any]] = None


def resolve_classes(
    query_class: Optional[str],
    available_classes: Sequence[str],
    matcher: TRTCLIPTextMatcher,
    *,
    top_k: int,
    min_similarity: float,
    margin: float,
) -> List[ResolvedClass]:
    query_class = _normalize_label(query_class)
    if not query_class or not available_classes:
        return []
    exact = [label for label in available_classes if _labels_match(query_class, label)]
    if exact:
        return [ResolvedClass(class_name=exact[0], similarity=1.0)]
    matches = matcher.top_matches(query_class, available_classes, top_k=top_k)
    if not matches:
        return []
    best = matches[0][1]
    resolved = []
    for class_name, similarity in matches:
        if similarity < min_similarity:
            continue
        if similarity + margin < best:
            continue
        resolved.append(ResolvedClass(class_name=class_name, similarity=similarity))
    if not resolved:
        class_name, similarity = matches[0]
        resolved = [ResolvedClass(class_name=class_name, similarity=similarity)]
    return resolved


def _score_attribute(attr: str, obj: Any, peer_group: Sequence[Any], scene_graph: SpatialSceneGraph, viewer_direction: str) -> Optional[float]:
    attr = _normalize_attribute_name(attr)
    if attr in IGNORED_ATTRS:
        return None
    try:
        index = next(i for i, item in enumerate(peer_group) if item.pred_id == obj.pred_id)
    except StopIteration:
        return None

    if attr == "upper":
        values = [item.center[2] for item in peer_group]
        return _soft_rank_score(values, index, lower_is_better=False, scale=scene_graph.vertical_scale)
    if attr == "lower":
        values = [item.center[2] for item in peer_group]
        return _soft_rank_score(values, index, lower_is_better=True, scale=scene_graph.vertical_scale)
    if attr == "largest":
        values = [item.volume for item in peer_group]
        return _soft_rank_score(values, index, lower_is_better=False, scale=max(scene_graph.horizontal_scale, 0.2))
    if attr == "smallest":
        values = [item.volume for item in peer_group]
        return _soft_rank_score(values, index, lower_is_better=True, scale=max(scene_graph.horizontal_scale, 0.2))
    if attr == "tallest":
        values = [float(item.size[2]) for item in peer_group]
        return _soft_rank_score(values, index, lower_is_better=False, scale=scene_graph.vertical_scale)
    if attr == "shortest":
        values = [float(item.size[2]) for item in peer_group]
        return _soft_rank_score(values, index, lower_is_better=True, scale=scene_graph.vertical_scale)
    if attr in {"leftmost", "rightmost", "frontmost", "backmost"}:
        front_axis, left_axis = _axis_from_viewer_direction(viewer_direction)
        centers = np.stack([item.center for item in peer_group], axis=0)
        if attr == "leftmost":
            values = [float(np.dot(center, left_axis)) for center in centers]
            return _soft_rank_score(values, index, lower_is_better=False, scale=scene_graph.align_scale)
        if attr == "rightmost":
            values = [float(np.dot(center, left_axis)) for center in centers]
            return _soft_rank_score(values, index, lower_is_better=True, scale=scene_graph.align_scale)
        if attr == "frontmost":
            values = [float(np.dot(center, front_axis)) for center in centers]
            return _soft_rank_score(values, index, lower_is_better=False, scale=scene_graph.align_scale)
        values = [float(np.dot(center, front_axis)) for center in centers]
        return _soft_rank_score(values, index, lower_is_better=True, scale=scene_graph.align_scale)
    if attr == "corner":
        dist, _ = scene_graph.nearest_corner_distance(obj)
        return _safe_exp_neg(dist, scene_graph.corner_scale)
    if attr == "center_of_room":
        return _safe_exp_neg(float(np.linalg.norm(obj.center - scene_graph.room_center)), scene_graph.midpoint_scale)
    if attr == "against_wall":
        return max(scene_graph.wall_affinity(obj).values())
    if attr == "on_wall":
        against = max(scene_graph.wall_affinity(obj).values())
        clearance = float(max(obj.box_min[2] - scene_graph.room_min[2], 0.0))
        elevated = 1.0 - _safe_exp_neg(clearance, scene_graph.vertical_scale)
        return _clamp01(against * elevated)
    return None


def _filter_ref_objects(
    objects: Sequence[Any],
    attributes: Sequence[str],
    scene_graph: SpatialSceneGraph,
    viewer_direction: str,
) -> Tuple[List[Any], List[str]]:
    filtered = list(objects)
    ignored: List[str] = []
    for attr in attributes:
        attr = _normalize_attribute_name(attr)
        if attr in IGNORED_ATTRS:
            ignored.append(attr)
            continue
        if attr not in FILTERABLE_REF_ATTRS:
            ignored.append(attr)
            continue
        if len(filtered) <= 1:
            break
        scored = []
        for obj in filtered:
            score = _score_attribute(attr, obj, filtered, scene_graph, viewer_direction)
            if score is not None:
                scored.append((obj, score))
        if not scored:
            ignored.append(attr)
            continue
        scored.sort(key=lambda item: item[1], reverse=True)
        keep = min(max(1, len(scored) // 2), 6)
        threshold = scored[keep - 1][1]
        filtered = [obj for obj, score in scored if score + 1e-6 >= threshold]
    return filtered or list(objects), ignored


def build_parse_prompt(description: str, class_inventory: Sequence[Mapping[str, Any]]) -> str:
    inventory_json = json.dumps(list(class_inventory), ensure_ascii=False, indent=2)
    return f"""You are a 3D referring-expression parser for a downstream query-graph grounder.
Your task is ONLY to convert the language description into a compact structured query graph.
Do NOT choose a predicted object id.

Available candidate class inventory for this scene:
{inventory_json}

Output JSON schema:
{{
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
open, closed, empty
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
  "parser_confidence": 0.92,
  "reason": "find the target cabinet right of the refrigerator",
  "nodes": [
    {{"id": "t", "role": "target", "class": "cabinet", "attributes": ["upper"]}},
    {{"id": "r1", "role": "ref", "class": "refrigerator", "attributes": []}}
  ],
  "relations": [
    {{"type": "right_of", "source": "t", "target": "r1", "weight": 1.0}}
  ]
}}

Target description: {description}
"""


def build_rerank_prompt(description: str, query_graph: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    compact = []
    for candidate in candidates:
        compact.append(
            {
                "pred_id": candidate.get("pred_id"),
                "class_name": candidate.get("class_name"),
                "center": candidate.get("center"),
                "total_score": candidate.get("total_score"),
                "best_direction": candidate.get("best_direction"),
                "constraint_scores": candidate.get("constraint_scores"),
                "post_attr_scores": candidate.get("post_attr_scores"),
            }
        )
    return f"""You are choosing the final target instance from a shortlist produced by a spatial graph matcher.
Choose exactly one pred_id from the shortlist.

Description:
{description}

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


def _chat_completion(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_body: Optional[Mapping[str, Any]] = None,
) -> str:
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra_body:
        request_kwargs["extra_body"] = dict(extra_body)
    response = client.chat.completions.create(**request_kwargs)
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content or ""
    return ""


def choose_reranked_pred_id(
    client: OpenAI,
    model: str,
    description: str,
    query_graph: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    extra_body: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[str]]:
    if not candidates:
        return None, None, None
    answer = _chat_completion(
        client,
        model=model,
        prompt=build_rerank_prompt(description, query_graph, candidates),
        max_tokens=256,
        extra_body=extra_body,
    )
    parsed = parse_response_json(answer)
    if not isinstance(parsed, dict):
        return None, None, answer.strip() or None
    try:
        pred_id = int(parsed.get("best_pred_id"))
    except (TypeError, ValueError):
        return None, parsed, answer.strip() or None
    shortlist_ids = {int(item["pred_id"]) for item in candidates if item.get("pred_id") is not None}
    if pred_id not in shortlist_ids:
        return None, parsed, answer.strip() or None
    return pred_id, parsed, answer.strip() or None


def build_scene_query(
    query_graph: QueryGraph,
    scene_graph: SpatialSceneGraph,
    matcher: TRTCLIPTextMatcher,
    viewer_direction: str,
    *,
    class_top_k: int,
    min_clip_similarity: float,
    class_margin: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[Any]]]:
    available_classes = sorted({_normalize_label(obj.class_name) for obj in scene_graph.objects if _normalize_label(obj.class_name)})
    node_map = query_graph.node_map()
    target_node = query_graph.target_node()

    resolved_classes: Dict[str, List[ResolvedClass]] = {}
    node_objects: Dict[str, List[Any]] = {}
    node_debug: Dict[str, Any] = {}

    for node in query_graph.nodes:
        resolved = resolve_classes(
            node.class_name,
            available_classes,
            matcher,
            top_k=class_top_k,
            min_similarity=min_clip_similarity,
            margin=class_margin,
        )
        resolved_classes[node.node_id] = resolved
        classes = [item.class_name for item in resolved]
        objects: List[Any] = []
        if classes:
            for class_name in classes:
                objects.extend(scene_graph.resolve_objects(class_name=class_name))
        if not objects and node.class_name:
            objects = scene_graph.resolve_objects(class_name=node.class_name)
        if not objects:
            objects = scene_graph.resolve_objects()

        ignored_attrs: List[str] = []
        if node.role == "ref" and node.attributes:
            objects, ignored_attrs = _filter_ref_objects(objects, node.attributes, scene_graph, viewer_direction)

        node_objects[node.node_id] = list(objects)
        node_debug[node.node_id] = {
            "requested_class": node.class_name,
            "resolved_classes": [vars(item) for item in resolved],
            "candidate_pred_ids": [obj.pred_id for obj in objects],
            "ignored_attributes": ignored_attrs,
        }

    target_resolved = resolved_classes[target_node.node_id]
    target_class = None
    if len(target_resolved) == 1:
        target_class = target_resolved[0].class_name
    elif not target_resolved:
        target_class = target_node.class_name

    query: Dict[str, Any] = {
        "target_class": target_class,
        "candidate_ids": [obj.pred_id for obj in node_objects[target_node.node_id]],
        "viewer_direction": viewer_direction,
        "fallback_to_all": True,
        "relations": [],
    }

    for attr in target_node.attributes:
        attr = _normalize_attribute_name(attr)
        if attr in TARGET_SCENE_ATTRS:
            query["relations"].append({"type": attr, "weight": 1.0})
        elif attr in IGNORED_ATTRS:
            node_debug[target_node.node_id].setdefault("ignored_attributes", []).append(attr)

    for relation in query_graph.relations:
        if relation.source != target_node.node_id:
            continue
        rel_payload: Dict[str, Any] = {"type": relation.rel_type, "weight": relation.weight}
        if relation.rel_type == "between" and len(relation.targets) >= 2:
            ref_a = node_objects.get(relation.targets[0], [])
            ref_b = node_objects.get(relation.targets[1], [])
            if ref_a:
                rel_payload["ref_a_pred_ids"] = [obj.pred_id for obj in ref_a]
            if ref_b:
                rel_payload["ref_b_pred_ids"] = [obj.pred_id for obj in ref_b]
            if not ref_a and relation.targets[0] in node_map:
                ref_node = node_map[relation.targets[0]]
                if ref_node.class_name:
                    rel_payload["ref_a_class"] = ref_node.class_name
            if not ref_b and relation.targets[1] in node_map:
                ref_node = node_map[relation.targets[1]]
                if ref_node.class_name:
                    rel_payload["ref_b_class"] = ref_node.class_name
            query["relations"].append(rel_payload)
            continue
        if not relation.targets:
            continue
        ref_node_id = relation.targets[0]
        ref_objs = node_objects.get(ref_node_id, [])
        if ref_objs:
            rel_payload["ref_pred_ids"] = [obj.pred_id for obj in ref_objs]
        else:
            ref_resolved = resolved_classes.get(ref_node_id, [])
            ref_classes = [item.class_name for item in ref_resolved]
            if len(ref_classes) == 1:
                rel_payload["ref_class"] = ref_classes[0]
            elif ref_classes:
                rel_payload["ref_classes"] = ref_classes
            elif ref_node_id in node_map and node_map[ref_node_id].class_name:
                rel_payload["ref_class"] = node_map[ref_node_id].class_name
        query["relations"].append(rel_payload)

    resolved_dict = {node_id: [vars(item) for item in items] for node_id, items in resolved_classes.items()}
    return query, {"resolved_classes": resolved_dict, "node_debug": node_debug}, node_objects


def apply_target_post_rerank(
    results: Sequence[Mapping[str, Any]],
    target_node: QueryNode,
    target_pool: Sequence[Any],
    scene_graph: SpatialSceneGraph,
    viewer_direction: str,
) -> List[Dict[str, Any]]:
    attrs = [_normalize_attribute_name(attr) for attr in target_node.attributes if _normalize_attribute_name(attr) in POST_RERANK_ATTRS]
    if not attrs:
        return [dict(item) for item in results]

    objects_by_id = {obj.pred_id: obj for obj in target_pool}
    updated: List[Dict[str, Any]] = []
    for result in results:
        item = dict(result)
        obj = objects_by_id.get(int(item["pred_id"])) or scene_graph.get_object(int(item["pred_id"]))
        attr_scores: Dict[str, float] = {}
        for attr in attrs:
            score = _score_attribute(attr, obj, target_pool, scene_graph, viewer_direction)
            if score is not None:
                attr_scores[attr] = float(score)
        if attr_scores:
            combined = _geometric_mean(attr_scores.values())
            item["total_score"] = float(item["total_score"]) * combined
            item["post_attr_scores"] = attr_scores
        updated.append(item)
    updated.sort(key=lambda item: float(item["total_score"]), reverse=True)
    return updated


def _compact_parse_log(query_graph: QueryGraph) -> str:
    target = query_graph.target_node()
    refs = [node.class_name for node in query_graph.nodes if node.role == "ref"]
    rels = [rel.rel_type for rel in query_graph.relations]
    return json.dumps(
        {
            "target": target.class_name,
            "target_attrs": list(target.attributes),
            "refs": refs,
            "relations": rels,
            "dir": query_graph.viewer_direction,
        },
        ensure_ascii=False,
    )


def _compact_ground_log(results: Sequence[Mapping[str, Any]]) -> str:
    if not results:
        return json.dumps({"best_pred_id": None}, ensure_ascii=False)
    best = results[0]
    return json.dumps(
        {
            "best_pred_id": best.get("pred_id"),
            "class": best.get("class_name"),
            "score": round(float(best.get("total_score", 0.0)), 4),
            "center": [round(float(x), 4) for x in best.get("center", [])],
        },
        ensure_ascii=False,
    )


class EdgeQueryGraphGrounder:
    def __init__(self, config: Optional[EdgeGroundingConfig] = None) -> None:
        self.config = config or EdgeGroundingConfig()
        self._client: Optional[OpenAI] = None
        self._matcher: Optional[TRTCLIPTextMatcher] = None

    def _resolve_pred_csv(self, pred_csv: Optional[Path]) -> Path:
        resolved = pred_csv or self.config.default_pred_csv
        if resolved is None:
            raise ValueError("pred_csv is required")
        return Path(resolved)

    def _resolve_scene_id(self, scene_id: Optional[str]) -> Optional[str]:
        return scene_id if scene_id is not None else self.config.default_scene_id

    def _load_records(self, pred_csv: Optional[Path], scene_id: Optional[str]) -> List[Dict[str, Any]]:
        return load_prediction_records_from_csv(
            self._resolve_pred_csv(pred_csv),
            scene_id_override=self._resolve_scene_id(scene_id),
        )

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = make_client(self.config.llm_base_url, self.config.llm_api_key)
        return self._client

    def _get_matcher(self) -> TRTCLIPTextMatcher:
        if self._matcher is None:
            self._matcher = TRTCLIPTextMatcher(
                self.config.clip_engine,
                self.config.bpe_path,
                output_name=self.config.engine_output_name,
                pooling=self.config.engine_output_pooling,
            )
        return self._matcher

    def _resolve_direction(self, query_graph: QueryGraph) -> str:
        effective_direction = (
            query_graph.viewer_direction
            if self.config.respect_llm_viewer_direction and query_graph.viewer_direction
            else _normalize_viewer_direction(self.config.viewer_direction)
        )
        if effective_direction is None:
            raise ValueError("Failed to determine viewer direction")
        return effective_direction

    def _parse_query_graph(
        self,
        describe: str,
        class_inventory: Sequence[Mapping[str, Any]],
        parse_json: Optional[str] = None,
    ) -> Tuple[QueryGraph, str]:
        if parse_json:
            parse_answer = parse_json
            parse_payload = parse_response_json(parse_json)
        else:
            parse_answer = _chat_completion(
                self._get_client(),
                model=self.config.llm_model,
                prompt=build_parse_prompt(describe, class_inventory),
                max_tokens=512,
                extra_body=self.config.llm_extra_body,
            )
            parse_payload = parse_response_json(parse_answer)
        if not isinstance(parse_payload, dict):
            raise ValueError(f"Parse stage failed: {parse_answer.strip() or 'empty response'}")
        return QueryGraph.from_payload(parse_payload), parse_answer

    def _run_from_records(
        self,
        describe: str,
        records: Sequence[Mapping[str, Any]],
        *,
        parse_json: Optional[str] = None,
        log: bool = False,
    ) -> Dict[str, Any]:
        class_inventory = build_class_inventory(records)
        scene_graph = SpatialSceneGraph.from_prediction_records(records)
        engine = SceneGraphQueryEngine(scene_graph)
        query_graph, parse_answer = self._parse_query_graph(describe, class_inventory, parse_json=parse_json)
        effective_direction = self._resolve_direction(query_graph)
        if log:
            print(f"[parse] {_compact_parse_log(query_graph)}")

        matcher = self._get_matcher()
        scene_query, class_debug, node_objects = build_scene_query(
            query_graph,
            scene_graph,
            matcher,
            effective_direction,
            class_top_k=self.config.class_top_k,
            min_clip_similarity=self.config.min_clip_similarity,
            class_margin=self.config.class_margin,
        )

        graph_result = engine.score_query(scene_query, top_k=self.config.top_k)
        target_node = query_graph.target_node()
        graph_result["results"] = apply_target_post_rerank(
            graph_result.get("results") or [],
            target_node,
            node_objects.get(target_node.node_id, []),
            scene_graph,
            effective_direction,
        )
        if log:
            print(f"[ground] {_compact_ground_log(graph_result.get('results') or [])}")

        final_best_pred_id: Optional[int] = None
        rerank_payload: Optional[Dict[str, Any]] = None
        rerank_raw: Optional[str] = None
        if graph_result.get("results"):
            final_best_pred_id = int(graph_result["results"][0]["pred_id"])

        if not self.config.disable_rerank and graph_result.get("results") and not parse_json:
            rerank_pred_id, rerank_payload, rerank_raw = choose_reranked_pred_id(
                self._get_client(),
                self.config.llm_model,
                describe,
                query_graph_to_dict(query_graph),
                graph_result["results"][: max(int(self.config.rerank_top_k), 1)],
                extra_body=self.config.llm_extra_body,
            )
            if rerank_pred_id is not None:
                final_best_pred_id = rerank_pred_id
            if log and rerank_payload is not None:
                print(
                    "[rerank] "
                    + json.dumps(
                        {
                            "best_pred_id": final_best_pred_id,
                            "confidence": rerank_payload.get("confidence"),
                        },
                        ensure_ascii=False,
                    )
                )

        final_object = scene_graph.get_object(final_best_pred_id) if final_best_pred_id is not None else None
        result_label = final_object.class_name if final_object is not None else None
        result_x = float(final_object.center[0]) if final_object is not None else None
        result_y = float(final_object.center[1]) if final_object is not None else None
        result_z = float(final_object.center[2]) if final_object is not None else None
        result_dict = {
            "id": final_best_pred_id,
            "label": result_label,
            "x": result_x,
            "y": result_y,
            "z": result_z,
        }
        return {
            "result": result_dict,
            "scene_id": records[0].get("scene_id") if records else None,
            "describe": describe,
            "viewer_direction": effective_direction,
            "query_graph": query_graph_to_dict(query_graph),
            "scene_query": scene_query,
            "resolved_classes": class_debug["resolved_classes"],
            "node_debug": class_debug["node_debug"],
            "graph_results": graph_result.get("results"),
            "parse_raw_response": parse_answer.strip(),
            "rerank_choice": rerank_payload,
            "rerank_raw_response": rerank_raw,
            "best_pred_id": final_best_pred_id,
            "best_class_name": result_label,
            "best_center": [result_x, result_y, result_z] if result_label is not None else None,
        }

    def run(
        self,
        describe: str,
        *,
        pred_csv: Optional[Path] = None,
        scene_id: Optional[str] = None,
        parse_json: Optional[str] = None,
        log: bool = False,
    ) -> Dict[str, Any]:
        records = self._load_records(pred_csv, scene_id)
        return self._run_from_records(describe, records, parse_json=parse_json, log=log)

    def find(
        self,
        describe: str,
        *,
        pred_csv: Optional[Path] = None,
        scene_id: Optional[str] = None,
        parse_json: Optional[str] = None,
        log: bool = False,
    ) -> Dict[str, Any]:
        payload = self.run(
            describe,
            pred_csv=pred_csv,
            scene_id=scene_id,
            parse_json=parse_json,
            log=log,
        )
        return payload["result"]

    def find_with_query_graph(
        self,
        describe: str,
        query_graph_payload: Mapping[str, Any],
        *,
        pred_csv: Optional[Path] = None,
        scene_id: Optional[str] = None,
        log: bool = False,
    ) -> Dict[str, Any]:
        records = self._load_records(pred_csv, scene_id)
        scene_graph = SpatialSceneGraph.from_prediction_records(records)
        engine = SceneGraphQueryEngine(scene_graph)
        query_graph = QueryGraph.from_payload(query_graph_payload)
        effective_direction = self._resolve_direction(query_graph)
        if log:
            print(f"[parse] {_compact_parse_log(query_graph)}")

        matcher = self._get_matcher()
        scene_query, _, node_objects = build_scene_query(
            query_graph,
            scene_graph,
            matcher,
            effective_direction,
            class_top_k=self.config.class_top_k,
            min_clip_similarity=self.config.min_clip_similarity,
            class_margin=self.config.class_margin,
        )

        graph_result = engine.score_query(scene_query, top_k=self.config.top_k)
        target_node = query_graph.target_node()
        graph_result["results"] = apply_target_post_rerank(
            graph_result.get("results") or [],
            target_node,
            node_objects.get(target_node.node_id, []),
            scene_graph,
            effective_direction,
        )
        if log:
            print(f"[ground] {_compact_ground_log(graph_result.get('results') or [])}")

        final_best_pred_id: Optional[int] = None
        if graph_result.get("results"):
            final_best_pred_id = int(graph_result["results"][0]["pred_id"])

        final_object = scene_graph.get_object(final_best_pred_id) if final_best_pred_id is not None else None
        return {
            "id": final_best_pred_id,
            "label": final_object.class_name if final_object is not None else None,
            "x": float(final_object.center[0]) if final_object is not None else None,
            "y": float(final_object.center[1]) if final_object is not None else None,
            "z": float(final_object.center[2]) if final_object is not None else None,
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge LLM + TRT-CLIP grounding from a local objects.csv")
    parser.add_argument("--description", default="find the chair farthest_from the desk", help="Referring expression to ground")
    parser.add_argument("--pred-csv", type=Path, default=DEFAULT_OBJECTS_PATH, help="Local CSV containing predicted objects")
    parser.add_argument("--scene-id", default=None, help="Optional scene id override")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")

    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL, help="Local OpenAI-compatible base URL")
    parser.add_argument("--llm-api-key", default=DEFAULT_LLM_API_KEY, help="API key for the local endpoint")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Model name exposed by the local endpoint")
    parser.add_argument("--llm-extra-body-json", default=None, help="Optional JSON object forwarded as extra_body")
    parser.add_argument("--parse-json", default=None, help="Optional inline parser JSON to skip the parse LLM call")

    parser.add_argument("--clip-engine", type=Path, default=DEFAULT_CLIP_PATH, help="TensorRT engine path for CLIP text encoder")
    parser.add_argument("--bpe-path", type=Path, default=DEFAULT_BPE_PATH, help="CLIP BPE vocab path")
    parser.add_argument("--engine-output-name", default=None, help="Optional TensorRT output binding name")
    parser.add_argument(
        "--engine-output-pooling",
        choices=("auto", "eot", "mean", "first"),
        default="auto",
        help="How to pool TensorRT output if it is [B, T, C]",
    )

    parser.add_argument("--viewer-direction", default=DEFAULT_VIEWER_DIRECTION, help="Single viewer direction used in grounding")
    parser.add_argument(
        "--respect-llm-viewer-direction",
        action="store_true",
        help="If set, use parser-returned viewer_direction when present",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many graph candidates to keep")
    parser.add_argument("--rerank-top-k", type=int, default=4, help="How many candidates to expose to the reranker")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable final LLM reranking")
    parser.add_argument("--class-top-k", type=int, default=3, help="Top-k CLIP class matches per query node")
    parser.add_argument("--min-clip-similarity", type=float, default=0.18)
    parser.add_argument("--class-margin", type=float, default=0.08)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if _normalize_viewer_direction(args.viewer_direction) is None:
        raise SystemExit(f"Unsupported --viewer-direction: {args.viewer_direction}")

    llm_extra_body = json.loads(args.llm_extra_body_json) if args.llm_extra_body_json else None
    grounder = EdgeQueryGraphGrounder(
        EdgeGroundingConfig(
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            clip_engine=args.clip_engine,
            bpe_path=args.bpe_path,
            default_pred_csv=args.pred_csv,
            default_scene_id=args.scene_id,
            engine_output_name=args.engine_output_name,
            engine_output_pooling=args.engine_output_pooling,
            viewer_direction=args.viewer_direction,
            respect_llm_viewer_direction=args.respect_llm_viewer_direction,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
            disable_rerank=args.disable_rerank,
            class_top_k=args.class_top_k,
            min_clip_similarity=args.min_clip_similarity,
            class_margin=args.class_margin,
            llm_extra_body=llm_extra_body,
        )
    )
    output = grounder.run(
        args.description,
        parse_json=args.parse_json,
        log=True,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(output["result"], ensure_ascii=False))


if __name__ == "__main__":
    main()
