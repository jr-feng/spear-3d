import sys
sys.path.append("third_party/FCGF")
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2
from Dataset.dataset import get_dataset_test
import tool.config as config
from Scene_rep import Scene_rep
import torch.nn.functional as F
from LLM import LLM
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from instance_tracker import InstanceTracker
from copy import deepcopy
from scene_generator import Scene_generator
import open3d as o3d
from typing import List, Tuple
current_dir = os.path.dirname(os.path.abspath(__file__))

maskclippp_path = os.path.join(current_dir, "third_party", "MaskCLIPpp")


if maskclippp_path not in sys.path:
    sys.path.insert(0, maskclippp_path)
import maskclippp
from maskclippp import add_maskformer2_config, add_maskclippp_config

predictor_path = os.path.join(maskclippp_path, "demo")
if predictor_path not in sys.path:
    sys.path.insert(0, predictor_path)
import predictor
from predictor import VisualizationDemo

import multiprocessing as mp
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskclippp_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.RUN_DEMO = True
    cfg.freeze()
    return cfg

def get_parser():

    parser = argparse.ArgumentParser(description="maskclippp demo for builtin configs")
    #maskclippp超参
    parser.add_argument(
        "--config-file",
        default="./third_party/MaskCLIPpp/configs/coco-stuff/eva-clip-vit-l-14-336/fcclip-l/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext_fcclip-l_ens.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--predefined-classes",
        type=str,
        default="coco2017|ade20k|lvis1203",
        help="The predefined classes for the model, Multiple classes are separated by '|'. Avaliable classes: coco2017, ade20k, lvis1203, cocostuff, ade847, ctx459, ctx59, voc20",
    )
    parser.add_argument(
        "--user-classes",
        type=str,
        default="chair|table|floor",
        help="Class labels defined by user. Different classes are separated by '|' and different synonyms of the same class are separated by ','. For example, 'tree,trees|sky,clouds'.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[
            "MODEL.WEIGHTS", "./third_party/MaskCLIPpp/output/ckpts/maskclippp/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext.pth",
            "MODEL.MASK_FORMER.TEST.PANOPTIC_ON", "True", 
            "MODEL.MASK_FORMER.TEST.INSTANCE_ON", "False",
            "MODEL.MASK_FORMER.TEST.SEMANTIC_ON", "False"
        ],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("-c", "--config", type=str, default="./config/scannet_test.yaml")
    # parser.add_argument("-d", "--dataset", type=str, default="./data/scans_OAS/scene0011_00")
    parser.add_argument("-d", "--dataset", type=str, default="./data_test/split")
    parser.add_argument("--seq_name", default="scene0011_00")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output_dir", default="./output_test/scannet")
    parser.add_argument(
        "-l", "--local_scene_files",
        nargs="+",
        type=str,
        default=["./data_test/update_output/update/seg_500.ply"],
        help="One or more local scene files or directories (can be multiple args or a single comma-separated string). Each entry should be a .ply (or a directory containing .ply files)."
    )

    # 输出可以是目录或文件路径，若用户不带扩展名会自动补 .ply
    parser.add_argument(
        "-u", "--updated_scene_output",
        type=str,
        default="./data_test/update_output",
        help="Output file path or directory. If a directory is provided, per-input outputs will be created inside it with .ply suffix."
    )


    return parser

def _color_from_global_id(global_id: int) -> np.ndarray:
    rng = np.random.default_rng(global_id)
    color = rng.uniform(0.2, 0.95, size=3)
    return color.astype(np.float32)


def _save_scene_instances(instances: List[Tuple[int, object]], output_path: str, voxel_size: float = 0.02):
    if not instances:
        return
    pts_list = []
    color_list = []
    for global_id, snap in instances:
        coords = snap.instance.mask_voxel_coords
        if coords is None or coords.shape[0] == 0:
            continue
        coords_np = coords.detach().cpu().numpy()
        colors_np = np.tile(_color_from_global_id(global_id), (coords_np.shape[0], 1))
        pts_list.append(coords_np)
        color_list.append(colors_np)
    if not pts_list:
        return
    pts = np.concatenate(pts_list, axis=0)
    colors = np.concatenate(color_list, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)

def _is_dataset_dir(path: str) -> bool:
    expected = ["color", "depth", "pose","intrinsic"]
    return all(os.path.isdir(os.path.join(path, name)) for name in expected)


def collect_dataset_paths(dataset_root: str):
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path {dataset_root} was not found.")
    dataset_root = os.path.abspath(dataset_root)
    if _is_dataset_dir(dataset_root):
        return [dataset_root]

    candidates = []
    for entry in sorted(os.listdir(dataset_root)):
        candidate = os.path.join(dataset_root, entry)
        if os.path.isdir(candidate) and _is_dataset_dir(candidate):
            candidates.append(candidate)
    return candidates if candidates else [dataset_root]


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    cfg_oas = config.load_config(args.config)

    mp.set_start_method("spawn", force=True)

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg_mask = setup_cfg(args)

    if len(args.predefined_classes) > 0:
            predefined_classes = args.predefined_classes.split("|")
    else:
            predefined_classes = []
    if len(args.user_classes) > 0:
            user_classes = args.user_classes.split("|")
    else:
            user_classes = []

    model = VisualizationDemo(
        cfg_mask, 
        predefined_classes=predefined_classes, 
        user_classes=user_classes,
        confidence_threshold=args.confidence_threshold
        )
 
    tracker = InstanceTracker(cfg_oas["scene"]["voxel_size"])######################

    dataset_paths = collect_dataset_paths(args.dataset)

    logger.info(f"Found {len(dataset_paths)} dataset(s) under {args.dataset}.")

    for dataset_path in dataset_paths:
        seq_name = args.seq_name if len(dataset_paths) == 1 else os.path.basename(os.path.normpath(dataset_path))
        logger.info(f"Processing dataset: {dataset_path} (seq_name={seq_name})")

        dataset = get_dataset_test(dataset_path, cfg_oas, args.device)

        seq_args = deepcopy(args)
        seq_args.dataset = dataset_path
        seq_args.seq_name = seq_name
        
        scene_rep = Scene_rep(cfg_oas, seq_args, dataset, args.device)

        scene_generator = Scene_generator(dataset, scene_rep, model, cfg_oas)

        inst_list, inst_text, inst_id, inst_center_coords, inst_coords, inst_indices, inst_sem_feature = scene_generator.generator()


        update_report = tracker.process_scene(inst_list, seq_name)########################

        for i in range(len(inst_text)):
            text = inst_text[i]
            id = inst_id[i]
            coords = inst_center_coords[i]
            print(f"instance_text_{i}:{text}")
            print(f"instance_id_{i}:{id}")
            print(f"instance_coords_{i}:{coords}")

###################
        if update_report.kept:
            print("Kept instances:")
            for global_id, snap in update_report.kept:
                print(f"  [keep] global_id={global_id}, semantic={snap.semantic}, center={snap.center}")
        if update_report.added:
            print("New instances:")
            for global_id, snap in update_report.added:
                print(f"  [add] global_id={global_id}, semantic={snap.semantic}, center={snap.center}")
        if update_report.removed:
            print("Removed instances:")
            for global_id, snap in update_report.removed:
                print(f"  [remove] global_id={global_id}, semantic={snap.semantic}, last_center={snap.center}")

    active_instances = [
        (gid, tracked.snapshot)
        for gid, tracked in tracker.global_instances.items()
        if tracked.active
    ]
    scene_voxel = cfg_oas["scene"]["voxel_size"]
    combined_output_path = os.path.join(args.updated_scene_output, "merged_instances.ply")
    _save_scene_instances(active_instances, combined_output_path, scene_voxel * 1.2)
    print("")
########################


            





