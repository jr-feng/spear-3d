from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from Instance import Instance


def _tensor_to_numpy(tensor) -> np.ndarray:
    if tensor is None:
        return np.empty((0, 3), dtype=np.float32)
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _compute_bbox(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if coords.size == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return zeros, zeros
    return coords.min(axis=0), coords.max(axis=0)


def _bbox_metrics(box_a: Tuple[np.ndarray, np.ndarray], box_b: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
    a_min, a_max = box_a
    b_min, b_max = box_b
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_extent = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_extent))
    if inter_vol <= 0.0:
        return 0.0, 0.0
    a_vol = float(np.prod(np.maximum(1e-6, a_max - a_min)))
    b_vol = float(np.prod(np.maximum(1e-6, b_max - b_min)))
    union = a_vol + b_vol - inter_vol
    iou = inter_vol / union if union > 1e-6 else 0.0
    overlap = inter_vol / min(a_vol, b_vol)
    return iou, overlap


def _bbox_volume(box: Tuple[np.ndarray, np.ndarray]) -> float:
    mins, maxs = box
    extent = np.maximum(0.0, maxs - mins)
    return float(np.prod(extent))


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    if denom <= 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


@dataclass
class InstanceSnapshot:
    instance: Instance
    semantic: str
    semantic_norm: str
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    num_points: int
    sem_feature: np.ndarray


@dataclass
class InstanceUpdateReport:
    scene_name: str
    kept: List[Tuple[int, InstanceSnapshot]] = field(default_factory=list)
    added: List[Tuple[int, InstanceSnapshot]] = field(default_factory=list)
    removed: List[Tuple[int, InstanceSnapshot]] = field(default_factory=list)


@dataclass
class TrackedInstance:
    snapshot: InstanceSnapshot
    miss_count: int = 0
    active: bool = True


class InstanceTracker:
    def __init__(
        self,
        voxel_size: float,
        center_thresh_multiplier: float = 6.0,
        iou_threshold: float = 0.2,
        overlap_threshold: float = 0.3,
        feature_similarity_thresh: float = 0.75,
        allowed_semantics: Optional[Sequence[str]] = None,
        removal_patience: int = 3,
        patch_feature_thresh: float = 0.9,
        patch_containment_thresh: float = 0.8,
    ) -> None:
        self.voxel_size = max(1e-3, float(voxel_size))
        self.center_thresh = max(self.voxel_size * center_thresh_multiplier, self.voxel_size * 2.0)
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.feature_similarity_thresh = feature_similarity_thresh
        self.allowed_semantics = (
            {sem.strip().lower() for sem in allowed_semantics} if allowed_semantics else None
        )
        self.removal_patience = max(1, int(removal_patience))
        self.patch_feature_thresh = patch_feature_thresh
        self.patch_containment_thresh = patch_containment_thresh

        self.global_instances: Dict[int, TrackedInstance] = {}
        self.next_global_id = 0

    def process_scene(self, instances: Iterable[Instance], scene_name: str) -> InstanceUpdateReport:
        report = InstanceUpdateReport(scene_name=scene_name)
        matched_ids: List[int] = []
        for inst in instances:
            snap = self._build_snapshot(inst)
            if snap is None:
                continue
            if self.allowed_semantics and snap.semantic_norm not in self.allowed_semantics:
                continue
            match_id = self._match_snapshot(snap)
            if match_id is None:
                patch_id = self._find_patch_host(snap)
                if patch_id is not None:
                    self._merge_snapshot(patch_id, snap)
                    matched_ids.append(patch_id)
                    tracked = self.global_instances[patch_id]
                    report.kept.append((patch_id, tracked.snapshot))
                else:
                    match_id = self._register_snapshot(snap)
                    report.added.append((match_id, snap))
                    matched_ids.append(match_id)
            else:
                tracked = self.global_instances[match_id]
                tracked.snapshot = snap
                tracked.miss_count = 0
                tracked.active = True
                matched_ids.append(match_id)
                report.kept.append((match_id, snap))

        for global_id, tracked in self.global_instances.items():
            if global_id in matched_ids or not tracked.active:
                continue
            tracked.miss_count += 1
            if tracked.miss_count >= self.removal_patience:
                tracked.active = False
                report.removed.append((global_id, tracked.snapshot))
        return report

    def _build_snapshot(self, inst: Instance) -> Optional[InstanceSnapshot]:
        coords = _tensor_to_numpy(inst.mask_voxel_coords)
        if coords.size == 0:
            return None
        semantic = inst.get_instance_text or ""
        semantic_norm = semantic.strip().lower()
        center = coords.mean(axis=0)
        bbox_min, bbox_max = _compute_bbox(coords)
        sem_feat = _tensor_to_numpy(inst.get_semantic_feature).astype(np.float32)
        return InstanceSnapshot(
            instance=inst,
            semantic=semantic,
            semantic_norm=semantic_norm,
            center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            num_points=coords.shape[0],
            sem_feature=sem_feat,
        )

    def _match_snapshot(self, snap: InstanceSnapshot) -> Optional[int]:
        best_id = None
        best_score = -np.inf
        for global_id, tracked in self.global_instances.items():
            if tracked is None or not tracked.active:
                continue
            feat_sim = _cosine_similarity(tracked.snapshot.sem_feature, snap.sem_feature)
            if feat_sim < self.feature_similarity_thresh:
                continue
            center_diff = float(np.linalg.norm(tracked.snapshot.center - snap.center))
            if center_diff > self.center_thresh:
                continue
            iou, overlap = _bbox_metrics(
                (tracked.snapshot.bbox_min, tracked.snapshot.bbox_max),
                (snap.bbox_min, snap.bbox_max),
            )
            if iou < self.iou_threshold and overlap < self.overlap_threshold:
                continue
            score = feat_sim + (1.0 / (1.0 + center_diff)) + iou + overlap
            if score > best_score:
                best_score = score
                best_id = global_id
        return best_id

    def _register_snapshot(self, snap: InstanceSnapshot) -> int:
        global_id = self.next_global_id
        self.next_global_id += 1
        self.global_instances[global_id] = TrackedInstance(snapshot=snap)
        return global_id

    def _find_patch_host(self, snap: InstanceSnapshot) -> Optional[int]:
        snap_vol = _bbox_volume((snap.bbox_min, snap.bbox_max))
        if snap_vol <= 1e-6:
            return None
        best_id = None
        best_score = -np.inf
        for global_id, tracked in self.global_instances.items():
            if not tracked.active:
                continue
            feat_sim = _cosine_similarity(tracked.snapshot.sem_feature, snap.sem_feature)
            if feat_sim < self.patch_feature_thresh:
                continue
            # center_diff = float(np.linalg.norm(tracked.snapshot.center - snap.center))
            # if center_diff > self.center_thresh:
            #     continue
            inter_min = np.maximum(tracked.snapshot.bbox_min, snap.bbox_min)
            inter_max = np.minimum(tracked.snapshot.bbox_max, snap.bbox_max)
            inter_extent = np.maximum(0.0, inter_max - inter_min)
            inter_vol = float(np.prod(inter_extent))
            if inter_vol <= 0.0:
                continue
            contain_ratio = inter_vol / max(1e-6, snap_vol)
            if contain_ratio < self.patch_containment_thresh:
                continue
            score = feat_sim + contain_ratio
            if score > best_score:
                best_score = score
                best_id = global_id
        return best_id

    def _merge_snapshot(self, host_id: int, snap: InstanceSnapshot) -> None:
        tracked = self.global_instances[host_id]
        host_inst = tracked.snapshot.instance
        coords_host = host_inst.mask_voxel_coords
        coords_new = snap.instance.mask_voxel_coords
        if coords_host is None or coords_host.shape[0] == 0:
            merged_coords = coords_new
        else:
            merged_coords = torch.cat([coords_host, coords_new], dim=0)
            merged_coords = torch.unique(merged_coords, dim=0)
        host_inst.mask_voxel_coords = merged_coords
        coords_np = merged_coords.detach().cpu().numpy()
        tracked.snapshot.center = coords_np.mean(axis=0)
        tracked.snapshot.bbox_min = coords_np.min(axis=0)
        tracked.snapshot.bbox_max = coords_np.max(axis=0)
        prev_pts = tracked.snapshot.num_points
        new_pts = snap.num_points
        tracked.snapshot.num_points = coords_np.shape[0]
        combined_feat = tracked.snapshot.sem_feature * prev_pts + snap.sem_feature * new_pts
        norm = np.linalg.norm(combined_feat) + 1e-8
        tracked.snapshot.sem_feature = combined_feat / norm


# from dataclasses import dataclass, field
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# import numpy as np
# import torch

# from Instance import Instance


# def _tensor_to_numpy(tensor) -> np.ndarray:
#     if tensor is None:
#         return np.empty((0, 3), dtype=np.float32)
#     if isinstance(tensor, torch.Tensor):
#         return tensor.detach().cpu().numpy()
#     return np.asarray(tensor)


# def _compute_bbox(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     if coords.size == 0:
#         zeros = np.zeros(3, dtype=np.float32)
#         return zeros, zeros
#     return coords.min(axis=0), coords.max(axis=0)


# def _bbox_metrics(box_a: Tuple[np.ndarray, np.ndarray], box_b: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
#     a_min, a_max = box_a
#     b_min, b_max = box_b
#     inter_min = np.maximum(a_min, b_min)
#     inter_max = np.minimum(a_max, b_max)
#     inter_extent = np.maximum(0.0, inter_max - inter_min)
#     inter_vol = float(np.prod(inter_extent))
#     if inter_vol <= 0.0:
#         return 0.0, 0.0
#     a_vol = float(np.prod(np.maximum(1e-6, a_max - a_min)))
#     b_vol = float(np.prod(np.maximum(1e-6, b_max - b_min)))
#     union = a_vol + b_vol - inter_vol
#     iou = inter_vol / union if union > 1e-6 else 0.0
#     overlap = inter_vol / min(a_vol, b_vol)
#     return iou, overlap


# def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
#     denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
#     if denom <= 0:
#         return 0.0
#     return float(np.dot(vec_a, vec_b) / denom)


# @dataclass
# class InstanceSnapshot:
#     instance: Instance
#     semantic: str
#     semantic_norm: str
#     center: np.ndarray
#     bbox_min: np.ndarray
#     bbox_max: np.ndarray
#     num_points: int
#     sem_feature: np.ndarray


# @dataclass
# class InstanceUpdateReport:
#     scene_name: str
#     kept: List[Tuple[int, InstanceSnapshot]] = field(default_factory=list)
#     added: List[Tuple[int, InstanceSnapshot]] = field(default_factory=list)
#     removed: List[Tuple[int, InstanceSnapshot]] = field(default_factory=list)


# @dataclass
# class TrackedInstance:
#     snapshot: InstanceSnapshot
#     miss_count: int = 0
#     active: bool = True


# class InstanceTracker:
#     def __init__(
#         self,
#         voxel_size: float,
#         center_thresh_multiplier: float = 6.0,
#         iou_threshold: float = 0.2,
#         overlap_threshold: float = 0.3,
#         feature_similarity_thresh: float = 0.7,
#         allowed_semantics: Optional[Sequence[str]] = None,
#         removal_patience: int = 3,
#     ) -> None:
#         self.voxel_size = max(1e-3, float(voxel_size))
#         self.center_thresh = max(self.voxel_size * center_thresh_multiplier, self.voxel_size * 2.0)
#         self.iou_threshold = iou_threshold
#         self.overlap_threshold = overlap_threshold
#         self.feature_similarity_thresh = feature_similarity_thresh

#         self.allowed_semantics = (
#             {sem.strip().lower() for sem in allowed_semantics} if allowed_semantics else None
#         )

#         self.removal_patience = max(1, int(removal_patience))

#         self.global_instances: Dict[int, TrackedInstance] = {}
#         self.next_global_id = 0

#     def process_scene(self, instances: Iterable[Instance], scene_name: str) -> InstanceUpdateReport:
#         report = InstanceUpdateReport(scene_name=scene_name)
#         matched_ids: List[int] = []
#         for inst in instances:
#             snap = self._build_snapshot(inst)
#             if snap is None:
#                 continue
#             if self.allowed_semantics and snap.semantic_norm not in self.allowed_semantics:
#                 continue
#             match_id = self._match_snapshot(snap)

#             if match_id is None:
#                 match_id = self._register_snapshot(snap)
#                 report.added.append((match_id, snap))
#                 matched_ids.append(match_id)
#             else:
#                 tracked = self.global_instances[match_id]
#                 tracked.snapshot = snap
#                 tracked.miss_count = 0
#                 tracked.active = True
#                 matched_ids.append(match_id)
#                 report.kept.append((match_id, snap))

#         for global_id, tracked in self.global_instances.items():
#             if global_id in matched_ids or not tracked.active:
#                 continue
#             tracked.miss_count += 1
#             if tracked.miss_count >= self.removal_patience:
#                 tracked.active = False
#                 report.removed.append((global_id, tracked.snapshot))
#         return report

#     def _build_snapshot(self, inst: Instance) -> Optional[InstanceSnapshot]:
#         coords = _tensor_to_numpy(inst.mask_voxel_coords)
#         if coords.size == 0:
#             return None
#         semantic = inst.get_instance_text or ""
#         semantic_norm = semantic.strip().lower()
#         center = coords.mean(axis=0)
#         bbox_min, bbox_max = _compute_bbox(coords)
#         sem_feat = _tensor_to_numpy(inst.get_semantic_feature).astype(np.float32)
#         return InstanceSnapshot(
#             instance=inst,
#             semantic=semantic,
#             semantic_norm=semantic_norm,
#             center=center,
#             bbox_min=bbox_min,
#             bbox_max=bbox_max,
#             num_points=coords.shape[0],
#             sem_feature=sem_feat,
#         )

#     def _match_snapshot(self, snap: InstanceSnapshot) -> Optional[int]:
#         best_id = None
#         best_score = -np.inf
#         for global_id, tracked in self.global_instances.items():
#             if tracked is None or not tracked.active:
#                 continue
#             feat_sim = _cosine_similarity(tracked.snapshot.sem_feature, snap.sem_feature)
#             if feat_sim < self.feature_similarity_thresh:
#                 continue

#             center_diff = float(np.linalg.norm(tracked.snapshot.center - snap.center))
#             if center_diff > self.center_thresh:
#                 continue

#             iou, overlap = _bbox_metrics(
#                 (tracked.snapshot.bbox_min, tracked.snapshot.bbox_max),
#                 (snap.bbox_min, snap.bbox_max),
#             )

#             if iou < self.iou_threshold and overlap < self.overlap_threshold:
#                 continue

#             score = feat_sim + (1.0 / (1.0 + center_diff)) + iou + overlap
#             if score > best_score:
#                 best_score = score
#                 best_id = global_id
#         return best_id

#     def _register_snapshot(self, snap: InstanceSnapshot) -> int:
#         global_id = self.next_global_id
#         self.next_global_id += 1
#         self.global_instances[global_id] = TrackedInstance(snapshot=snap)
#         return global_id



