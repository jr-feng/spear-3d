import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
from natsort import natsorted
import json

from tool.geometric_helpers import compose_transformations

import sys
import argparse
import multiprocessing as mp
import tempfile
import time
import warnings
import tqdm
from detectron2.data.detection_utils import read_image




def get_dataset(data_location, instance_dir, cfg, device="cuda:0"):
    if cfg["dataset"] == "scannet":
        dataset = ScannetDataset
    if cfg["dataset"] == "scenenn":
        dataset = SceneNNDataset
    if cfg["dataset"] == "my_dataset":
        dataset = MyDataset

    return dataset(data_location, instance_dir, cfg, device)

def get_dataset_test(data_location, cfg, device="cuda:0", subsample=None):
    if cfg["dataset"] == "scannet_test":
        dataset = ScannetDataset_test
    if cfg["dataset"] == "scenenn_test":
        dataset = SceneNNDataset_test
    
    return dataset(data_location, cfg, device, subsample=subsample)

import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import open3d as o3d
import glob
from natsort import natsorted

# class SceneNNDataset_test(Dataset):
#     def __init__(self, data_location, cfg, device="cuda:0"):
#         self.cfg = cfg
#         self.data_location = data_location
#         self.seq_name = self.data_location.split("/")[-1]
#         self.color_dir = os.path.join(self.data_location, "image")
#         self.color_basename_list = natsorted(os.listdir(self.color_dir))
#         self.color_basename_list = [n for n in self.color_basename_list if n.lower().endswith(".png")]
#         self.device = device
#         self.target_h = cfg["cam"]["img_h"]
#         self.target_w = cfg["cam"]["img_w"]
#         self.depth_scale = cfg["cam"]["depth_scale"]
#         self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
#         self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
#         intrinsic_file = os.path.join(self.data_location, "intrinsic", "intrinsic_depth.txt")
#         cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
#         self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
#         self.frame_num = self.get_scene_pose_num()
#         #######333
#         self.last_seg_frame_id = self.get_last_seg_frame_id(self.cfg["mapping"]["keyframe_freq"])
#         self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
#         self.bbox = None
#         self.gt_ply_file = self.find_gt_ply(self.data_location)
#         if self.cfg["cam"]["bound"]:
#             self.load_bound()
#         self.min_max_xyz = self.get_bbox(self.gt_ply_file)
#         self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj


#     def get_scene_pose_num(self):
#         pose_list = os.listdir(os.path.join(self.data_location, "pose"))
#         pose_list = natsorted(pose_list)
#         return len(pose_list)

#     def __len__(self):
#         return len(self.color_basename_list)

#     def find_gt_ply(self, dir):
#         gt_ply_files = glob.glob( os.path.join(dir, "*.ply") )
#         if len(gt_ply_files) == 1:
#             return gt_ply_files[0]
#         else:
#             return None

#     def get_bbox(self, ply_file):
#         if ply_file is None or not os.path.exists(ply_file):
#             min_max_xyz = [[0., 10.], [0., 10.], [0., 5.]]
#         else:
#             pc = o3d.io.read_point_cloud(ply_file)
#             min_xyz = pc.get_min_bound().astype("float32")
#             max_xyz = pc.get_max_bound().astype("float32")
#             min_max_xyz = np.stack([min_xyz, max_xyz], axis=-1).tolist()
#         return min_max_xyz

#     def get_last_seg_frame_id(self, seg_interval=0):
#         if seg_interval <= 0:
#             seg_interval = self.cfg["seg"]["seg_add_interval"]

#         seg_frame_ids = [ int(color_base_name[5:-4]) for color_base_name in self.color_basename_list]
#         last_seg_frame_id = seg_frame_ids[0]
#         for seg_frame_id in seg_frame_ids[::-1]:
#             if (seg_frame_id - 1) % seg_interval == 0:
#                 last_seg_frame_id = seg_frame_id
#                 break
#         return last_seg_frame_id

#         # @brief: load bounding box of GT mesh
#     def load_bound(self):
#         gt_mesh_path = self.gt_ply_file
#         if gt_mesh_path is not None and os.path.exists(gt_mesh_path):
#             gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
#             self.bbox = gt_mesh.get_oriented_bounding_box()  # open3d.geometry.OrientedBoundingBox obj

#     def load_poses(self):
#         pose_map = {}
#         pose_dir = os.path.join(self.data_location, "pose")
#         posefiles = natsorted(glob.glob(os.path.join(pose_dir, "*.txt")))
#         if not posefiles:
#             posefiles = natsorted(glob.glob(os.path.join(pose_dir, "*.npy")))
#         for posefile in posefiles:
#             if posefile.endswith(".npy"):
#                 _pose_np = np.load(posefile).astype("float32")
#             else:
#                 _pose_np = np.loadtxt(posefile).astype("float32")
#             frame_id = os.path.splitext(os.path.basename(posefile))[0]
#             pose_map[frame_id] = torch.from_numpy(_pose_np)
#         if len(pose_map) == 0:
#             raise FileNotFoundError(f"No pose files found under {pose_dir} (expected .txt or .npy)")
#         return pose_map

#     def get_poses(self, relative=False):
#         pose_map = self.load_poses()  # dict: frame_id -> Tensor(4, 4)
#         depth_dir = os.path.join(self.data_location, "depth")
#         depth_ids = set()
#         if os.path.isdir(depth_dir):
#             for name in os.listdir(depth_dir):
#                 if name.lower().endswith(".png") and name.startswith("depth"):
#                     depth_ids.add(name[5:-4])

#         valid_color_list = []
#         poses = []
#         for color_base in self.color_basename_list:
#             frame_id = color_base[5:-4]
#             if frame_id in pose_map and frame_id in depth_ids:
#                 valid_color_list.append(color_base)
#                 poses.append(pose_map[frame_id])

#         if not valid_color_list:
#             raise FileNotFoundError(
#                 f"No matching frames across image/depth/pose under {self.data_location}"
#             )

#         self.color_basename_list = valid_color_list
#         poses = torch.stack(poses, dim=0).to(self.device)
#         self.first_pose_c2w = poses[0]
#         if relative:  # default
#             pose_first = poses[0]  # Tensor(4, 4)
#             pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
#             final_poses = compose_transformations(pose_first_inv, poses)
#         else:
#             final_poses = poses
#         return final_poses

#     def get_intrinsics(self, intrinsic_mat):
#         intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
#         intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
#         return intrinsic_cam_parameters


#     def __getitem__(self, index):
#         color_base = self.color_basename_list[index]
#         frame_id = color_base[5:-4]
#         color_img_path = os.path.join(self.data_location, "image", color_base)
#         depth_img_path = os.path.join(self.data_location, "depth", f"depth{frame_id}.png")


#         # Step 2: load color/depth image
#         depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
#         if depth_img is None:
#             raise FileNotFoundError(f"Failed to read depth image: {depth_img_path}")
#         depth_img = depth_img.astype("float32") / self.depth_scale

#         color_img = cv2.imread(color_img_path)
#         if color_img is None:
#             raise FileNotFoundError(f"Failed to read color image: {color_img_path}")
#         color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB

#         if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
#             color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

#         color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(h, w, 3), [0, 1], dtype=float32
#         depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

#         # depth clipping
#         if self.depth_near > 0 and self.depth_far > 0:
#             depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
#             depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))


#         pose_matrix = self.poses[index]  # Tensor(4, 4)
#         return color_img, color_img_path, depth_img, pose_matrix


class SceneNNDataset_test(Dataset):
    def __init__(self, data_location, cfg, device="cuda:0"):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-1]
        # print(self.seq_name)
        # print(self.data_location)
        self.color_dir = os.path.join(self.data_location, "image")
        self.pose_dir = os.path.join(self.data_location, "pose")
        self.color_basename_list = natsorted(os.listdir(self.color_dir))
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
        intrinsic_file = os.path.join(self.data_location, "intrinsic", "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
        # self.valid_ids = sorted([int(f[5:-4]) for f in os.listdir(self.color_dir) if f.endswith(".png")])
        pose_ids = sorted([int(f[:-4]) for f in os.listdir(self.pose_dir) if f.endswith(".txt")])
        self.valid_ids, self.bad_ids = self._filter_valid_ids(pose_ids)
        self.color_basename_list = [f"image{frame_id:05d}.png" for frame_id in self.valid_ids]
        self.frame_num = self.get_scene_pose_num()
        self.last_seg_frame_id = self.get_last_seg_frame_id(self.cfg["mapping"]["keyframe_freq"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None
        self.gt_ply_file = self.find_gt_ply(self.data_location)
        if self.cfg["cam"]["bound"]:
            self.load_bound()
        self.min_max_xyz = self.get_bbox(self.gt_ply_file)
        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj


    def _filter_valid_ids(self, pose_ids):
        valid_ids = []
        bad_ids = []
        for frame_id in pose_ids:
            color_img_path = os.path.join(self.data_location, "image", f"image{frame_id:05d}.png")
            depth_img_path = os.path.join(self.data_location, "depth", f"depth{frame_id:05d}.png")
            if not os.path.isfile(color_img_path) or not os.path.isfile(depth_img_path):
                bad_ids.append(frame_id)
                continue
            color_img = cv2.imread(color_img_path)
            if color_img is None:
                bad_ids.append(frame_id)
                continue
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                bad_ids.append(frame_id)
                continue
            valid_ids.append(frame_id)
        if bad_ids:
            print(
                f"[SceneNNDataset_test] Skipping {len(bad_ids)} frames with missing/corrupt files in"
                f" {self.seq_name} (e.g., {bad_ids[0]})."
            )
        return valid_ids, bad_ids

    def get_scene_pose_num(self):
        pose_list = os.listdir(os.path.join(self.data_location, "pose"))
        pose_list = natsorted(pose_list)
        return len(pose_list)

    def __len__(self):
        # return self.frame_num
        return len(self.valid_ids)

    def find_gt_ply(self, dir):
        gt_ply_files = glob.glob( os.path.join(dir, "*.ply") )
        if len(gt_ply_files) == 1:
            return gt_ply_files[0]
        else:
            return None

    def get_bbox(self, ply_file):
        if ply_file is None or not os.path.exists(ply_file):
            min_max_xyz = [[0., 10.], [0., 10.], [0., 5.]]
        else:
            pc = o3d.io.read_point_cloud(ply_file)
            min_xyz = pc.get_min_bound().astype("float32")
            max_xyz = pc.get_max_bound().astype("float32")
            min_max_xyz = np.stack([min_xyz, max_xyz], axis=-1).tolist()
        return min_max_xyz

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(color_base_name[5:-4]) for color_base_name in self.color_basename_list]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if seg_frame_id % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

        # @brief: load bounding box of GT mesh
    def load_bound(self):
        gt_mesh_path = self.gt_ply_file
        if gt_mesh_path is not None and os.path.exists(gt_mesh_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
            self.bbox = gt_mesh.get_oriented_bounding_box()  # open3d.geometry.OrientedBoundingBox obj

    def load_poses(self):
        poses = []
        posefiles = natsorted( glob.glob( os.path.join(self.data_location, "pose/*.txt") ) )
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters


    def __getitem__(self, index):
        # color_img_path = os.path.join(self.data_location, "color", "%d.jpg" % index)
        # depth_img_path = os.path.join(self.data_location, "depth", "%d.png" % index)
        real_id = self.valid_ids[index]
        color_img_path = os.path.join(self.data_location, "image", f"image{real_id:05d}.png")
        depth_img_path = os.path.join(self.data_location, "depth", f"depth{real_id:05d}.png")


        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB
        
   
        if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)


        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))


        pose_matrix = self.poses[index]  # Tensor(4, 4)
        return color_img, color_img_path, depth_img, pose_matrix

class ScannetDataset(Dataset):
    def __init__(self, data_location, instance_dir, cfg, device="cuda:0"):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-1]
        self.instance_dir = instance_dir
        self.mask_image_dir = os.path.join(self.instance_dir, "mask")
        self.mask_embed_dir = os.path.join(self.instance_dir, "mask_embeddings")
        self.mask_basename_list = natsorted( os.listdir(self.mask_image_dir) )
        self.mask_embed_basename_list = natsorted( os.listdir(self.mask_embed_dir) )
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
        intrinsic_file = os.path.join(self.data_location, "intrinsic", "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)

        
        self.frame_num = self.get_scene_img_num()
        self.last_seg_frame_id = self.get_last_seg_frame_id(cfg["seg"]["seg_add_interval"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None
        if self.cfg["cam"]["bound"]:
            self.load_bound()
        self.gt_ply_file = self.find_gt_ply( os.path.join(data_location, "../") )
        self.min_max_xyz = self.get_bbox(self.gt_ply_file)

        # crop image edge
        self.h_crop = cfg["cam"]["h_crop"]
        self.w_crop = cfg["cam"]["w_crop"]
        if self.h_crop > 0 and self.w_crop > 0:
            self.target_h -= 2 * self.h_crop
            self.target_w -= 2 * self.w_crop
            self.cam_intrinsic[0, 2] -= self.w_crop
            self.cam_intrinsic[1, 2] -= self.h_crop

        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj


    def get_scene_img_num(self):
        color_img_list = os.listdir(os.path.join(self.data_location, "color"))
        color_img_list.sort(key=lambda x:int(x[:-4]))
        return len(color_img_list)

    def __len__(self):
        return self.frame_num

    def find_gt_ply(self, dir):
        gt_ply_files = glob.glob( os.path.join(dir, "*_vh_clean_2.ply") )
        if len(gt_ply_files) == 1:
            return gt_ply_files[0]
        else:
            return None

    def get_bbox(self, ply_file):
        if ply_file is None or not os.path.exists(ply_file):
            min_max_xyz = [[0., 10.], [0., 10.], [0., 5.]]
        else:
            pc = o3d.io.read_point_cloud(ply_file)
            min_xyz = pc.get_min_bound().astype("float32")
            max_xyz = pc.get_max_bound().astype("float32")
            min_max_xyz = np.stack([min_xyz, max_xyz], axis=-1).tolist()
        return min_max_xyz

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(mask_base_name[:-4]) for mask_base_name in self.mask_basename_list ]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if seg_frame_id % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

    # @brief: load bounding box of GT mesh
    def load_bound(self):
        gt_mesh_path = os.path.join(self.data_location, "../", "%s_vh_clean_2.ply" % self.seq_name)
        if os.path.exists(gt_mesh_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
            self.bbox = gt_mesh.get_oriented_bounding_box()  # open3d.geometry.OrientedBoundingBox obj

    def load_poses(self):
        poses = []
        posefiles = natsorted( glob.glob( os.path.join(self.data_location, "pose/*.txt") ) )
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters

    def __getitem__(self, index):
        color_img_path = os.path.join(self.data_location, "color", "%d.jpg" % index)
        depth_img_path = os.path.join(self.data_location, "depth", "%d.png" % index)

        # Step 1: load segmentation image (and semantic feature of each detected mask)
        mask_image_basename = "%d.png" % index
        mask_embed_basename = "%d.pt" % index
        if mask_image_basename in self.mask_basename_list and mask_embed_basename in self.mask_embed_basename_list:
            instance_path = os.path.join(self.mask_image_dir, mask_image_basename)
            segmentation = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)  # ndarray(H, W), dtype=uint8
            segmentation = torch.from_numpy(segmentation).to(self.device)
            if self.h_crop > 0 and self.w_crop > 0:
                segmentation = segmentation[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]
            self.latest_seg_img = segmentation

            instance_embed_path = os.path.join(self.mask_embed_dir, mask_embed_basename)
            mask_embeddings = torch.load(instance_embed_path).to(self.device)
            seg_flag = True
        else:
            segmentation = torch.zeros_like(self.latest_seg_img).to(self.device)
            mask_embeddings = torch.zeros((1, self.cfg["mask"]["feature_dim"])).to(self.device)
            seg_flag = False

        # Step 2: load color/depth image
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB
        # color_img = cv2.imread(color_img_path)
        if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # color_img = color_img[:, :, ::-1]  # BGR2RGB

        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))

        if self.h_crop > 0 and self.w_crop > 0:
            color_img = color_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop, :]
            depth_img = depth_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]

        pose_matrix = self.poses[index]  # Tensor(4, 4)
        return color_img, depth_img, pose_matrix, segmentation, mask_embeddings, seg_flag



class MyDataset(Dataset):
    def __init__(self, data_location,instance_dir,cfg, device="cuda:0"):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-2]
        self.instance_dir = instance_dir
        self.mask_image_dir = os.path.join(self.instance_dir, "mask")
        self.mask_embed_dir = os.path.join(self.instance_dir, "mask_embeddings")
        self.mask_basename_list = natsorted( os.listdir(self.mask_image_dir) )
        self.mask_embed_basename_list = natsorted( os.listdir(self.mask_embed_dir) )
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1
        intrinsic_file = os.path.join(self.data_location, "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
        self.frame_num = self.get_scene_img_num()
        self.last_seg_frame_id = self.get_last_seg_frame_id(cfg["seg"]["seg_add_interval"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None


        self.h_crop = cfg["cam"]["h_crop"]
        self.w_crop = cfg["cam"]["w_crop"]
        if self.h_crop > 0 and self.w_crop > 0:
            self.target_h -= 2 * self.h_crop
            self.target_w -= 2 * self.w_crop
            self.cam_intrinsic[0, 2] -= self.w_crop
            self.cam_intrinsic[1, 2] -= self.h_crop

        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj
        self.rgb_files, self.depth_files = self.get_rgbd()
        





    def get_scene_img_num(self):
        color_img_list = os.listdir(os.path.join(self.data_location, "color"))
        color_img_list = natsorted(color_img_list)
        print(len(color_img_list))
        return len(color_img_list)

    def __len__(self):
        return self.frame_num

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(mask_base_name[:4]) for mask_base_name in self.mask_basename_list ]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if seg_frame_id % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

    def load_poses(self):
        poses = []
        posefiles = natsorted( glob.glob( os.path.join(self.data_location, "poses/*.txt") ) )
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters

    def get_rgbd(self):
        rgb_dir = os.path.join(self.data_location, "color")
        depth_dir = os.path.join(self.data_location, "depth")
        rgb_files = natsorted( os.listdir(rgb_dir) )
        depth_files = natsorted( os.listdir(depth_dir) )
        return rgb_files, depth_files

    def __getitem__(self, index):
        color_img_path = self.rgb_files[index]
        depth_img_path = self.depth_files[index]

        # Step 1: load segmentation image (and semantic feature of each detected mask)
        mask_image_basename = "%d.png" % index
        mask_embed_basename = "%d.pt" % index
        if mask_image_basename in self.mask_basename_list and mask_embed_basename in self.mask_embed_basename_list:
            instance_path = os.path.join(self.mask_image_dir, mask_image_basename)
            segmentation = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)  # ndarray(H, W), dtype=uint8
            segmentation = torch.from_numpy(segmentation).to(self.device)
            if self.h_crop > 0 and self.w_crop > 0:
                segmentation = segmentation[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]
            self.latest_seg_img = segmentation

            instance_embed_path = os.path.join(self.mask_embed_dir, mask_embed_basename)
            mask_embeddings = torch.load(instance_embed_path).to(self.device)
            seg_flag = True
        else:
            segmentation = torch.zeros_like(self.latest_seg_img).to(self.device)
            mask_embeddings = torch.zeros((1, self.cfg["mask"]["feature_dim"])).to(self.device)
            seg_flag = False

        # Step 2: load color/depth image
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        predictions, visualized_output, m_emds = self.demo.run_on_image(color_img)

        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB


        if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))

        if self.h_crop > 0 and self.w_crop > 0:
            color_img = color_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop, :]
            depth_img = depth_img[self.h_crop:-self.h_crop, self.w_crop:-self.w_crop]

        pose_matrix = self.poses[index]  # Tensor(4, 4)
        return color_img, depth_img, pose_matrix ,m_emds

class ScannetDataset_test(Dataset):
    def __init__(self, data_location,cfg, device="cuda:0", subsample=None):
        self.cfg = cfg
        self.data_location = data_location
        self.seq_name = self.data_location.split("/")[-1]
        self.color_dir = os.path.join(self.data_location, "color")
        self.depth_dir = os.path.join(self.data_location, "depth")
        self.pose_dir = os.path.join(self.data_location, "pose")
        self.device = device
        self.target_h = cfg["cam"]["img_h"]
        self.target_w = cfg["cam"]["img_w"]
        
        self.depth_scale = cfg["cam"]["depth_scale"]
        self.depth_near = cfg["cam"]["depth_near"] if cfg["cam"]["depth_near"] > 0 else -1
        self.depth_far = cfg["cam"]["depth_far"] if cfg["cam"]["depth_far"] > 0 else -1

        intrinsic_file = os.path.join(self.data_location, "intrinsics", "intrinsic_depth.txt")
        cam_intrinsic = np.loadtxt(intrinsic_file)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
        self.cam_intrinsic = torch.from_numpy(cam_intrinsic).to(self.device)
        self.valid_ids, self.bad_ids = self._filter_valid_ids()
        if subsample is not None and 0 < int(subsample) < len(self.valid_ids):
            # evenly subsample N frames (OVI-MAP-style 200-frame trajectory alignment)
            n = int(subsample)
            n_orig = len(self.valid_ids)
            keep = np.round(np.linspace(0, n_orig - 1, n)).astype(int)
            self.valid_ids = [self.valid_ids[i] for i in keep]
            print(f"[dataset] {self.seq_name}: evenly subsample {n}/{n_orig} frames")
        self.color_basename_list = [f"{frame_id}.jpg" for frame_id in self.valid_ids]
        self.frame_num = len(self.valid_ids)
        self.last_seg_frame_id = self.get_last_seg_frame_id(cfg["seg"]["seg_add_interval"])
        self.poses = self.get_poses()  # default: relative poses to first pose, Tensor(n, 4, 4)
        self.bbox = None
        if self.cfg["cam"]["bound"]:
            self.load_bound()
        self.gt_ply_file = self.find_gt_ply( os.path.join(data_location, "../") )
        self.min_max_xyz = self.get_bbox(self.gt_ply_file)


        self.pinhole_cam_intrinsic = self.get_intrinsics(self.cam_intrinsic.cpu().numpy())  # o3d.camera.PinholeCameraIntrinsic obj


    def get_scene_img_num(self):
        if hasattr(self, "valid_ids"):
            return len(self.valid_ids)
        color_img_list = os.listdir(os.path.join(self.data_location, "color"))
        color_img_list.sort(key=lambda x:int(x[:-4]))
        return len(color_img_list)

    def __len__(self):
        # return self.frame_num
        return len(self.valid_ids)

    def find_gt_ply(self, dir):
        gt_ply_files = glob.glob( os.path.join(dir, "*_vh_clean_2.ply") )
        if len(gt_ply_files) == 1:
            return gt_ply_files[0]
        else:
            return None

    def get_bbox(self, ply_file):
        if ply_file is None or not os.path.exists(ply_file):
            min_max_xyz = [[0., 10.], [0., 10.], [0., 5.]]
        else:
            pc = o3d.io.read_point_cloud(ply_file)
            min_xyz = pc.get_min_bound().astype("float32")
            max_xyz = pc.get_max_bound().astype("float32")
            min_max_xyz = np.stack([min_xyz, max_xyz], axis=-1).tolist()
        return min_max_xyz

    def get_last_seg_frame_id(self, seg_interval=0):
        if seg_interval <= 0:
            seg_interval = self.cfg["seg"]["seg_add_interval"]

        seg_frame_ids = [ int(color_base_name[:-4]) for color_base_name in self.color_basename_list ]
        last_seg_frame_id = seg_frame_ids[0]
        for seg_frame_id in seg_frame_ids[::-1]:
            if seg_frame_id % seg_interval == 0:
                last_seg_frame_id = seg_frame_id
                break
        return last_seg_frame_id

    # @brief: load bounding box of GT mesh
    def load_bound(self):
        gt_mesh_path = os.path.join(self.data_location, "../", "%s_vh_clean_2.ply" % self.seq_name)
        if os.path.exists(gt_mesh_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
            self.bbox = gt_mesh.get_oriented_bounding_box()  # open3d.geometry.OrientedBoundingBox obj

    def load_poses(self):
        poses = []
        posefiles = [os.path.join(self.pose_dir, f"{frame_id}.txt") for frame_id in self.valid_ids]
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
            poses.append(_pose)
        poses = torch.stack(poses, dim=0).to(self.device)
        return poses

    def _list_frame_ids(self, dir_path, suffix):
        frame_ids = []
        for name in os.listdir(dir_path):
            if not name.endswith(suffix):
                continue
            stem = name[:-len(suffix)]
            try:
                frame_id = int(stem)
            except ValueError:
                continue
            frame_ids.append(frame_id)
        return sorted(frame_ids)

    def _filter_valid_ids(self):
        color_ids = self._list_frame_ids(self.color_dir, ".jpg")
        depth_ids = self._list_frame_ids(self.depth_dir, ".png")
        pose_ids = self._list_frame_ids(self.pose_dir, ".txt")
        valid_ids = sorted(set(color_ids) & set(depth_ids) & set(pose_ids))
        bad_ids = sorted((set(color_ids) | set(depth_ids) | set(pose_ids)) - set(valid_ids))
        if bad_ids:
            print(
                f"[ScannetDataset_test] Skipping {len(bad_ids)} frames with missing files in"
                f" {self.seq_name} (e.g., {bad_ids[0]})."
            )
        return valid_ids, bad_ids

    def get_poses(self, relative=False):
        poses = self.load_poses()  # Tensor(n, 4, 4)
        self.first_pose_c2w = poses[0]
        if relative:  # default
            pose_first = poses[0]  # Tensor(4, 4)
            pose_first_inv = torch.inverse(pose_first).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # Tensor(n, 4, 4)
            final_poses = compose_transformations(pose_first_inv, poses)
        else:
            final_poses = poses
        return final_poses

    def get_intrinsics(self, intrinsic_mat):
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(self.target_w, self.target_h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
        return intrinsic_cam_parameters

    def __getitem__(self, index):
        # color_img_path = os.path.join(self.data_location, "color", "%d.jpg" % index)
        # depth_img_path = os.path.join(self.data_location, "depth", "%d.png" % index)
        real_id = self.valid_ids[index]
        color_img_path = os.path.join(self.data_location, "color", "%d.jpg" % real_id)
        depth_img_path = os.path.join(self.data_location, "depth", "%d.png" % real_id)


        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype("float32") / self.depth_scale

        color_img = cv2.imread(color_img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB
        
   
        if color_img.shape[0] != depth_img.shape[0] or color_img.shape[1] != depth_img.shape[1]:
            color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)


        color_img = torch.from_numpy(color_img).to(self.device) / 255  # Tensor(j, w, 3), [0, 1], dtype=float32
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Tensor(h, w)

        # depth clipping
        if self.depth_near > 0 and self.depth_far > 0:
            depth_mask = ( (depth_img > self.depth_near) & (depth_img < self.depth_far) )
            depth_img = torch.where(depth_mask, depth_img, torch.zeros_like(depth_img))


        pose_matrix = self.poses[index]  # Tensor(4, 4)
        return color_img, color_img_path, depth_img, pose_matrix
