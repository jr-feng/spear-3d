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
import logging
from Dataset.dataset import get_dataset_test
import tool.config as config
from Scene_rep import Scene_rep
import torch.nn.functional as F
from LLM import LLM
from scene_update import SceneUpdater, load_point_cloud
import open3d as o3d
import multiprocessing as mp
from sam3_api import sam3_api


def setup_simple_logger(output_dir):
    logger = logging.getLogger("sam3_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"))
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(description="Batch evaluate scenes with SAM3 panoptic segmentation + OnlineAnySeg fusion")
    parser.add_argument("--input", nargs="+", help="Unused placeholder (kept for compatibility)")
    parser.add_argument("-c", "--config", type=str, default="./config/sceneNN_test.yaml")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("-o", "--output_dir", default="./output_test/sceneNN_sam3_test")
    parser.add_argument("--scans-root", type=str, default="./data_eval/scenenn", help="Root directory containing scene folders to evaluate")
    parser.add_argument("--merge-gpu", type=int, default=1, help="GPU id for OnlineAnySeg fusion (e.g., 1)")
    parser.add_argument("--failed-log", type=str, default="./output_test/sceneNN_sam3_test/failed_scenes.log", help="Where to save failed scene list")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional specific scene names to run (space- or comma-separated). If omitted, all scenes under scans-root are processed.",
    )
    return parser

def normalize_scene_names(scene_args):
    if not scene_args:
        return None
    names = []
    for item in scene_args:
        for part in item.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names or None

def run_single_scene(scene_dir, base_args, merge_device, failed_scenes, scene_fps, logger):
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

        intr_path = os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
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
        total_api_call_time = 0.0  # 总的 API 请求耗时（包含传输）
        total_model_latency = 0.0  # API 返回的纯模型推理耗时
        api_calls = 0

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
                    color_img_path = str(color_img_path[0])
                    color_mask_img = cv2.imread(color_img_path)
                    if color_mask_img is None:
                        raise FileNotFoundError(f"Failed to read image: {color_img_path}")
                    color_mask_img_re = cv2.resize(color_mask_img, (640, 480), interpolation=cv2.INTER_AREA)

                    api_start = time.time()
                    panoptic_map, segments_info, segments_count,latency_sec = sam3_api("scenenn", scene_name, color_mask_img_re)
                    api_duration = time.time() - api_start
                    total_api_call_time += api_duration
                    total_model_latency += latency_sec
                    api_calls += 1
                    target_device = scene_rep.device
                    seg_image = torch.from_numpy(panoptic_map).to(device=target_device)

                    if segments_info is not None and segments_count > 0:
                        for i in range(segments_count):
                            mask_id = segments_info[i]['instance_id']
                            mask_class_emd = segments_info[i]['clip_embedding']
                            mask_text = segments_info[i]['category_name']

                            if mask_id is None or mask_class_emd is None:
                                continue
                            mask_ids.append(mask_id)
                            mask_features.append(torch.as_tensor(mask_class_emd, dtype=torch.float32, device=target_device))
                            mask_texts.append(mask_text)
                        if len(mask_features) == 0:
                            seg_flag = True
                            continue
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
        pred_instance_mask_list = scene_rep.save_ckpt(frame_id_final, ckpt_save_dir, filter_flag=True, ckpt_path=final_ckpt_path, save_colors=True)

        scene_rep.save_merging_result(frame_id_final, reextract=False)

        final_pc_path = os.path.join(local_output_dir, "final.ply")
        scene_rep.save_pc(scene_rep.points, scene_rep.colors, final_pc_path)
        end_time = time.time()
        elapsed = max(end_time - start_time, 1e-6)
        transfer_overhead = max(total_api_call_time - total_model_latency, 0.0)
        adjusted_elapsed = max(elapsed - transfer_overhead, 1e-6)
        fps = processed_frames / adjusted_elapsed
        fps_msg = (
            f"Scene {scene_name} finished successfully. FPS (exclude API transfer): {fps:.2f} "
            f"({processed_frames} frames in {adjusted_elapsed:.2f}s, removed {transfer_overhead:.2f}s transfer over {api_calls} calls)"
        )
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
    fps_log_path = os.path.join(output_dir, "scene_fps.txt")
    with open(fps_log_path, "w") as f:
        f.write("Scene FPS results (API transfer excluded):\n")
        for name, fps in scene_fps:
            f.write(f"{name}: {fps:.4f} FPS\n")
        f.write(f"Average FPS: {avg_fps:.4f}\n")
    print(f"已将 FPS 结果写入专用文件 {fps_log_path}")


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
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_simple_logger(args.output_dir)
    logger.info("Arguments: " + str(args))

    merge_device = f"cuda:{args.merge_gpu}"

    failed_scenes = []
    candidate_names = normalize_scene_names(args.scenes)
    if not candidate_names:
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
        ckpt_final_path = os.path.join(args.output_dir, scene_name, "ckpt_final.npz")
        ckpt_final_dir = os.path.join(args.output_dir, scene_name, "ckpt_final")
        if os.path.isfile(ckpt_final_path) or os.path.isdir(ckpt_final_dir):
            found_path = ckpt_final_path if os.path.isfile(ckpt_final_path) else ckpt_final_dir
            msg = f"[Skip] {scene_name}: ckpt_final exists at {found_path}"
            print(msg)
            failed_scenes.append((scene_name, "ckpt_final exists"))
            if logger:
                logger.info(msg)
            continue
        run_single_scene(scene_dir, args, merge_device, failed_scenes, scene_fps, logger)

    if scene_fps:
        avg_fps = sum(fps for _, fps in scene_fps) / len(scene_fps)
        print("\n=== 场景帧率汇总（已去除 API 传输时间） ===")
        for name, fps in scene_fps:
            print(f"{name}: {fps:.2f} FPS")
        print(f"平均帧率: {avg_fps:.2f} FPS")
        write_fps_log(scene_fps, args.output_dir, avg_fps)
        append_fps_to_eval_log(args.output_dir, avg_fps, logger)
    else:
        logger.info("No scene FPS recorded.")

    write_failed_log(failed_scenes, args.failed_log)


if __name__ == "__main__":
    main()
