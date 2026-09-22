import sys
sys.path.append("third_party/FCGF")
import argparse
import os
import traceback
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2
import time
from Dataset.dataset import get_dataset_test
import tool.config as config
from Scene_rep import Scene_rep
import torch.nn.functional as F
from LLM import LLM
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from scene_update import SceneUpdater, load_point_cloud
import open3d as o3d
import multiprocessing as mp


current_dir = os.path.dirname(os.path.abspath(__file__))
maskclippp_path = os.path.join(current_dir, "third_party", "MaskCLIPpp")
predictor_path = os.path.join(maskclippp_path, "demo")

if maskclippp_path not in sys.path:
    sys.path.insert(0, maskclippp_path)
if predictor_path not in sys.path:
    sys.path.insert(0, predictor_path)

import maskclippp
from maskclippp import add_maskformer2_config, add_maskclippp_config
import predictor
from predictor import VisualizationDemo


def setup_cfg(args):
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
    parser = argparse.ArgumentParser(description="Batch evaluate scenes with MaskCLIP++ + OnlineAnySeg")
    parser.add_argument(
        "--config-file",
        default="./third_party/MaskCLIPpp/configs/coco-stuff/eva-clip-vit-l-14-336/fcclip-l/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext_fcclip-l_ens.yaml",
        metavar="FILE",
        help="path to MaskCLIP++ config file",
    )
    parser.add_argument("--input", nargs="+", help="Unused placeholder (kept for compatibility)")
    parser.add_argument(
        "--predefined-classes",
        type=str,
        default="coco2017|ade20k|lvis1203",
        help="Predefined classes for MaskCLIP++ model",
    )
    parser.add_argument(
        "--user-classes",
        type=str,
        default="chair|table|floor",
        help="User defined class labels, separated by '|'",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.18,
        help="Minimum score for instance predictions to be used",
    )
    parser.add_argument(
        "--opts",
        help="Modify MaskCLIP++ config options using 'KEY VALUE' pairs",
        default=[
            "MODEL.WEIGHTS", "./third_party/MaskCLIPpp/output/ckpts/maskclippp/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext.pth",
            "MODEL.MASK_FORMER.TEST.PANOPTIC_ON", "True",
            "MODEL.MASK_FORMER.TEST.INSTANCE_ON", "False",
            "MODEL.MASK_FORMER.TEST.SEMANTIC_ON", "False",
        ],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("-c", "--config", type=str, default="./config/scannet_test.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output_dir", default="./output_test/scannet")
    parser.add_argument(
        "-l",
        "--local_scene_files",
        nargs="+",
        type=str,
        default=["./data_test/update_output/update/seg_500.ply"],
        help="One or more local scene files or directories (can be multiple args or a single comma-separated string). Each entry should be a .ply (or a directory containing .ply files).",
    )
    parser.add_argument(
        "-u",
        "--updated_scene_output",
        type=str,
        default="./data_test/update_output/update.ply",
        help="Output file path or directory. If a directory is provided, per-input outputs will be created inside it with .ply suffix.",
    )
    parser.add_argument("--scans-root", type=str, default="./data_eval/scans", help="Root directory containing scene folders to evaluate")
    parser.add_argument("--mask-gpu", type=int, default=0, help="GPU id for MaskCLIP++ inference (e.g., 0)")
    parser.add_argument("--merge-gpu", type=int, default=0, help="GPU id for OnlineAnySeg fusion (e.g., 1)")
    parser.add_argument("--failed-log", type=str, default="./output_test/scannet/failed_scenes.log", help="Where to save failed scene list")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional specific scene names to run. If omitted, all scenes under scans-root are processed.",
    )
    return parser


def prepare_maskclip_cfg(args, mask_device):
    cfg_mask = setup_cfg(args)
    cfg_mask.defrost()
    cfg_mask.MODEL.DEVICE = mask_device
    cfg_mask.freeze()
    return cfg_mask


def run_single_scene(scene_dir, base_args, merge_device, failed_scenes, scene_fps, logger, demo):
    scene_name = os.path.basename(os.path.normpath(scene_dir))
    start_msg = f"========== Start scene: {scene_name} =========="
    print(f"\n{start_msg}")
    if logger:
        logger.info(start_msg)
    try:
        args = deepcopy(base_args)
        args.dataset = scene_dir
        args.seq_name = scene_name
        args.device = merge_device
        if torch.cuda.is_available():
            torch.cuda.set_device(args.merge_gpu)

        intr_path = os.path.join(scene_dir, "intrinsics", "intrinsic_depth.txt")
        if not os.path.isfile(intr_path):
            failed_scenes.append((scene_name, f"{intr_path} not found"))
            print(f"[Skip] {scene_name}: {intr_path} not found")
            return

        cfg_oas = config.load_config(args.config)

        dataset = get_dataset_test(args.dataset, cfg_oas, args.device)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        scene_rep = Scene_rep(cfg_oas, args, dataset, args.device)
        last_valid_c2w = torch.eye(4)
        seg_flag = False
        processed_frames = 0
        start_time = time.time()

        local_output_dir = os.path.join(args.output_dir, args.seq_name)
        os.makedirs(local_output_dir, exist_ok=True)

        max_frame_num = dataset.last_seg_frame_id

        for frame_id, (color_img, color_img_path, depth_img, pose_c2w) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=scene_name,
        ):
            if max_frame_num is not None and frame_id >= max_frame_num:
                break
            processed_frames += 1

            if torch.isnan(pose_c2w[0]).any().item() or torch.isinf(pose_c2w[0]).any().item():
                pose_c2w[0] = last_valid_c2w
            else:
                last_valid_c2w = pose_c2w[0]

            if frame_id % cfg_oas["mapping"]["keyframe_freq"] == 0:
                with torch.no_grad():
                    pose_w2c = torch.inverse(pose_c2w[0])
                    frustum_block_coords, extrinsic = scene_rep.integrate_frame(frame_id, color_img[0], depth_img[0], pose_w2c)

                    mask_ids = []
                    mask_features = []
                    mask_texts = []
                    color_img_path = color_img_path[0]
                    color_mask_img = read_image(color_img_path)
                    color_mask_img_re = cv2.resize(color_mask_img, (640, 480), interpolation=cv2.INTER_AREA)

                    predictions, _ = demo.run_on_image(color_mask_img_re)
                    seg_image = predictions["panoptic_seg"][0]
                    if predictions["panoptic_seg"][1] is not None and len(predictions["panoptic_seg"][1]) > 0:
                        for i in range(len(predictions["panoptic_seg"][1])):
                            mask_id = predictions["panoptic_seg"][1][i]["id"]
                            mask_class_emd = predictions["panoptic_seg"][1][i]["embed"]
                            mask_text = predictions["panoptic_seg"][1][i]["text"]

                            if mask_id is None or mask_class_emd is None:
                                continue
                            mask_ids.append(mask_id)
                            mask_features.append(mask_class_emd)
                            first_text = mask_text.split(",")[0].strip()
                            mask_texts.append(first_text)

                        masks_features = torch.stack(mask_features, dim=0)
                    else:
                        seg_flag = True
                        continue

                    if (frame_id % cfg_oas["seg"]["seg_add_interval"] == 0 or seg_flag) and len(mask_features) != 0:
                        valid_mask_ids, valid_mask_voxels = scene_rep.insert_seg_frame(
                            frame_id,
                            color_img[0],
                            depth_img[0],
                            pose_c2w[0],
                            frustum_block_coords,
                            seg_image,
                            mask_features,
                            mask_texts,
                        )
                        seg_flag = False
                        if frame_id > 0 and frame_id % scene_rep.merge_frame_interval == 0:
                            scene_rep.update_masks(frame_id)

            if cfg_oas["save"]["ckpt_interval"] > 0 and frame_id > 0 and frame_id % cfg_oas["save"]["ckpt_interval"] == 0:
                if scene_rep.vis_pc_flag:
                    scene_rep.vis_pc.update()

                ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id)
                os.makedirs(ckpt_save_dir, exist_ok=True)
                scene_rep.save_ckpt(frame_id, ckpt_save_dir)

        print("################ Begin to save final ckpt and seg mesh/pointcloud...")
        frame_id_final = dataset.__len__() - 1 if max_frame_num is None else max_frame_num - 1

        ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id_final)
        os.makedirs(ckpt_save_dir, exist_ok=True)

        final_ckpt_path = os.path.join(local_output_dir, "ckpt_final.npz")
        pred_instance_mask_list = scene_rep.save_ckpt(frame_id_final, ckpt_save_dir, filter_flag=True, ckpt_path=final_ckpt_path)

        scene_rep.save_merging_result(frame_id_final, reextract=False)

        final_pc_path = os.path.join(local_output_dir, "final.ply")
        scene_rep.save_pc(scene_rep.points, scene_rep.colors, final_pc_path)
        end_time = time.time()
        elapsed = max(end_time - start_time, 1e-6)
        fps = processed_frames / elapsed
        fps_msg = f"Scene {scene_name} finished successfully. FPS: {fps:.2f} ({processed_frames} frames in {elapsed:.2f}s)"
        print(fps_msg)
        if logger:
            logger.info(fps_msg)
        scene_fps.append((scene_name, fps))
    except Exception as exc:
        failed_scenes.append((scene_name, str(exc)))
        print(f"[Failed] Scene {scene_name}: {exc}")
        if logger:
            logger.error(f"[Failed] Scene {scene_name}: {exc}")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()


def write_failed_log(failed_scenes, log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w") as f:
        if not failed_scenes:
            f.write("All scenes processed successfully.\n")
            print("所有场景运行成功。")
            return

        f.write("Failed or skipped scenes:\n")
        for scene_name, reason in failed_scenes:
            f.write(f"{scene_name}: {reason}\n")
        print(f"发现 {len(failed_scenes)} 个异常/跳过场景，已写入日志 {log_path}")


def write_fps_log(scene_fps, output_dir, avg_fps):
    if not scene_fps:
        return
    os.makedirs(output_dir, exist_ok=True)
    fps_log_path = os.path.join(output_dir, "scene_fps.log")
    with open(fps_log_path, "w") as f:
        f.write("Scene FPS results:\n")
        for name, fps in scene_fps:
            f.write(f"{name}: {fps:.4f} FPS\n")
        f.write(f"Average FPS: {avg_fps:.4f}\n")
    print(f"已将 FPS 结果写入 {fps_log_path}")


def append_fps_to_eval_log(output_dir, avg_fps, logger=None):
    """Append average FPS to eval_log.csv as the last line."""
    eval_log_path = os.path.join(output_dir, "eval_log.csv")
    if not os.path.isfile(eval_log_path):
        msg = f"eval_log.csv not found at {eval_log_path}, skip appending FPS."
        print(msg)
        if logger:
            logger.warning(msg)
        return
    try:
        with open(eval_log_path, "a") as f:
            f.write(f"average_fps,{avg_fps:.4f},,\n")
        msg = f"已将平均 FPS 追加到 {eval_log_path}"
        print(msg)
        if logger:
            logger.info(msg)
    except Exception as exc:
        msg = f"追加 FPS 到 {eval_log_path} 失败: {exc}"
        print(msg)
        if logger:
            logger.error(msg)


def main():
    parser = get_parser()
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(name="fvcore", output=args.output_dir)
    logger = setup_logger(output=args.output_dir)
    logger.info("Arguments: " + str(args))

    mask_device = f"cuda:{args.mask_gpu}"
    merge_device = f"cuda:{args.merge_gpu}"

    if len(args.predefined_classes) > 0:
        predefined_classes = args.predefined_classes.split("|")
    else:
        predefined_classes = []
    if len(args.user_classes) > 0:
        user_classes = args.user_classes.split("|")
    else:
        user_classes = []

    cfg_mask = prepare_maskclip_cfg(args, mask_device)
    demo = VisualizationDemo(
        cfg_mask,
        predefined_classes=predefined_classes,
        user_classes=user_classes,
        confidence_threshold=args.confidence_threshold,
    )

    failed_scenes = []
    if args.scenes:
        candidate_names = args.scenes
    else:
        candidate_names = [
            d
            for d in sorted(os.listdir(args.scans_root))
            if os.path.isdir(os.path.join(args.scans_root, d))
        ]

    scene_paths = []
    for s in candidate_names:
        full = os.path.join(args.scans_root, s)
        if os.path.isdir(full):
            scene_paths.append(full)
        else:
            failed_scenes.append((s, "scene directory not found"))
            print(f"[Skip] {s}: directory not found")

    scene_fps = []
    for scene_dir in tqdm(scene_paths, desc="Scenes", unit="scene"):
        scene_name = os.path.basename(os.path.normpath(scene_dir))
        # final_ply_path = os.path.join(args.output_dir, scene_name, "final.ply")
        # if os.path.isfile(final_ply_path):
        #     print(f"[Skip] {scene_name}: found existing result at {final_ply_path}")
        #     failed_scenes.append((scene_name, f"skipped: existing result at {final_ply_path}"))
        #     continue
        run_single_scene(scene_dir, args, merge_device, failed_scenes, scene_fps, logger, demo)

    if scene_fps:
        avg_fps = sum(fps for _, fps in scene_fps) / len(scene_fps)
        print("\n=== 场景帧率汇总 ===")
        for name, fps in scene_fps:
            print(f"{name}: {fps:.2f} FPS")
        print(f"平均帧率: {avg_fps:.2f} FPS")
        logger.info("=== 场景帧率汇总 ===")
        for name, fps in scene_fps:
            logger.info(f"{name}: {fps:.2f} FPS")
        logger.info(f"平均帧率: {avg_fps:.2f} FPS")
        write_fps_log(scene_fps, args.output_dir, avg_fps)
        append_fps_to_eval_log(args.output_dir, avg_fps, logger)
    else:
        logger.info("No scene FPS recorded.")

    write_failed_log(failed_scenes, args.failed_log)


if __name__ == "__main__":
    main()
