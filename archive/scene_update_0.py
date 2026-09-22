import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import open3d as o3d


def _to_numpy(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    if colors.size == 0:
        return colors
    if colors.max() > 1.0:
        return colors / 255.0
    return colors


@dataclass
class VoxelInfo:
    color: np.ndarray
    count: int = 1


@dataclass
class SceneState:
    voxel_size: float
    voxels: Dict[Tuple[int, int, int], VoxelInfo]

    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.voxels:
            return None
        coords = np.array(list(self.voxels.keys()), dtype=np.int64)
        return coords.min(axis=0), coords.max(axis=0)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.voxels:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        coords = []
        colors = []
        for key, info in self.voxels.items():
            coords.append((np.array(key, dtype=np.float32) + 0.5) * self.voxel_size)
            colors.append(info.color.astype(np.float32))
        return np.stack(coords), np.stack(colors)


@dataclass
class SceneUpdateSummary:
    added: int = 0
    removed: int = 0
    updated: int = 0

    def __str__(self) -> str:
        return f"added={self.added}, removed={self.removed}, updated={self.updated}"


def build_scene_state(points: np.ndarray, colors: Optional[np.ndarray], voxel_size: float) -> SceneState:
    pts = _to_numpy(points)
    if pts is None or pts.size == 0:
        return SceneState(voxel_size=voxel_size, voxels={})
    pts = pts.astype(np.float32)
    color_array = _to_numpy(colors)
    if color_array is None or color_array.shape[0] != pts.shape[0]:
        color_array = np.ones_like(pts, dtype=np.float32)
    color_array = color_array.astype(np.float32)
    color_array = _normalize_colors(color_array)

    voxel_coords = np.floor(pts / voxel_size).astype(np.int64)
    voxels: Dict[Tuple[int, int, int], VoxelInfo] = {}
    for idx, coord in enumerate(voxel_coords):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        color = color_array[idx]
        if key not in voxels:
            voxels[key] = VoxelInfo(color=color.copy(), count=1)
        else:
            info = voxels[key]
            next_count = info.count + 1
            info.color = (info.color * info.count + color) / next_count
            info.count = next_count
            voxels[key] = info
    return SceneState(voxel_size=voxel_size, voxels=voxels)


def _dilate_keys(keys: set, radius: int) -> set:
    if radius <= 0:
        return set(keys)
    offsets = range(-radius, radius + 1)
    dilated = set()
    for key in keys:
        x, y, z = key
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    dilated.add((x + dx, y + dy, z + dz))
    return dilated


class SceneUpdater:
    def __init__(
        self,
        voxel_size: float,
        remove_obsolete: bool = False,
        overlap_margin_voxels: int = 0,
    ):
        """
        Args:
            voxel_size: Edge length of each voxel cell.
            remove_obsolete: If True, any global voxels that previously overlapped the local scan
                but are absent in the new observation will be deleted (local map takes precedence).
            overlap_margin_voxels: Expand the overlap region by this many voxel units before
                pruning, which can compensate for noisy localization.
        """
        self.voxel_size = voxel_size
        self.remove_obsolete = remove_obsolete
        self.overlap_margin_voxels = max(0, int(overlap_margin_voxels))
        self.global_scene: Optional[SceneState] = None

    def initialize_global_scene(self, points: np.ndarray, colors: Optional[np.ndarray]) -> SceneState:
        self.global_scene = build_scene_state(points, colors, self.voxel_size)
        return self.global_scene

    def update_with_local_scene(self, local_points: np.ndarray, local_colors: Optional[np.ndarray]) -> SceneUpdateSummary:
        if self.global_scene is None:
            raise RuntimeError("Global scene is not initialized.")
        local_state = build_scene_state(local_points, local_colors, self.voxel_size)
        return self._apply_local_state(local_state)

    def _apply_local_state(self, local_state: SceneState) -> SceneUpdateSummary:
        if self.global_scene is None:
            raise RuntimeError("Global scene is not initialized.")

        global_voxels = self.global_scene.voxels
        local_voxels = local_state.voxels

        global_keys = set(global_voxels.keys())
        print("global_key:",len(global_keys))
        local_keys = set(local_voxels.keys())
        print("local_key:",len(local_keys))

        removed = 0
        if self.remove_obsolete:
            influence_keys = _dilate_keys(local_keys, self.overlap_margin_voxels)
            remove_keys = {key for key in influence_keys if key in global_keys and key not in local_keys}
            for key in remove_keys:
                global_voxels.pop(key, None)
            removed = len(remove_keys)
            global_keys.difference_update(remove_keys)

        add_keys = local_keys - global_keys
        update_keys = global_keys & local_keys

        for key in add_keys:
            global_voxels[key] = local_voxels[key]

        for key in update_keys:
            g_info = global_voxels[key]
            l_info = local_voxels[key]
            total_count = g_info.count + l_info.count
            if total_count == 0:
                continue
            blended_color = (g_info.color * g_info.count + l_info.color * l_info.count) / total_count
            global_voxels[key] = VoxelInfo(color=blended_color, count=total_count)

        return SceneUpdateSummary(added=len(add_keys), removed=removed, updated=len(update_keys))

    def export_scene_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.global_scene is None:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        return self.global_scene.as_arrays()


def load_point_cloud(path: str) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.size == 0 and points.size != 0:
        colors = np.ones_like(points, dtype=np.float32)
    colors = _normalize_colors(colors)
    return points, colors


