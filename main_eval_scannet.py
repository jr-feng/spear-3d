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
from scene_update import SceneUpdater, load_point_cloud
import open3d as o3d
import multiprocessing as mp
from sam3_api import sam3_api
from clip_features import CLIPFeatureExtractor


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
    parser.add_argument("-c", "--config", type=str, default=None, help="config yaml (default: scannet_test_1024.yaml for --arm A/B, else scannet_test.yaml)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output_dir", default=None, help="output dir (default: output_ral/<arm> when --arm given, else ./output_test/nr3d)")
    parser.add_argument("--scans-root", type=str, default="/workspace/nr3d/scans", help="Root directory containing scene folders to evaluate")
    parser.add_argument("--merge-gpu", type=int, default=0, help="GPU id for OnlineAnySeg fusion (e.g., 1)")
    parser.add_argument("--failed-log", type=str, default=None, help="Where to save failed scene list (default: <output_dir>/failed_scenes.log)")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional specific scene names to run. If omitted, all scenes under scans-root are processed.",
    )
    parser.add_argument(
        "--arm",
        default=None,
        choices=["A", "B", "C", "D"],
        help="2x2 ablation arm (SAM3 masks only, via this script): "
             "A = SAM3 + label-bound text (feature-source=text), B = SAM3 + per-crop CLIP (feature-source=crop); "
             "output saved to output_ral/<arm>. "
             "C/D need CropFormer masks and must be run with the CropFormer pipeline (scripts/mask_predict + main.py), not this script.",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Evenly subsample to N frames per scene (OVI-MAP 200-frame alignment; None = all frames)",
    )
    parser.add_argument(
        "--dense-seg", action="store_true",
        help="Segmentation on EVERY processed frame (keyframe_freq=1, seg_add_interval=1). "
             "Use with --subsample 200 to match OVI-MAP's per-frame segmentation on 200-frame trajectories",
    )
    parser.add_argument(
        "--feature-source",
        default="server",
        choices=["server", "text", "crop"],
        help="server: use the server's label text embedding (default, 256-dim); "
             "text: local CLIP ViT-H-14 text embedding (1024-dim, arm A); "
             "crop: local per-crop CLIP ViT-H-14 image embedding (1024-dim, arm B). "
             "text/crop require config feature_dim=1024 (see config/scannet_test_1024.yaml).",
    )
    return parser

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

        intr_path = os.path.join(scene_dir, "intrinsics", "intrinsic_depth.txt")
        if not os.path.isfile(intr_path):
            failed_scenes.append((scene_name, f"{intr_path} not found"))
            print(f"[Skip] {scene_name}: {intr_path} not found")
            return

        cfg_oas = config.load_config(args.config)
        if args.dense_seg:
            cfg_oas["mapping"]["keyframe_freq"] = 1
            cfg_oas["seg"]["seg_add_interval"] = 1
            print(f"[dense-seg] keyframe_freq/seg_add_interval -> 1 (per-frame segmentation)")

        dataset = get_dataset_test(args.dataset, cfg_oas, args.device, subsample=args.subsample)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        scene_rep = Scene_rep(cfg_oas, args, dataset, args.device)
        clip_fe = CLIPFeatureExtractor(args.device) if args.feature_source in ("text", "crop") else None
        last_valid_c2w = torch.eye(4)
        seg_flag = False
        processed_frames = 0
        start_time = time.time()
        total_api_call_time = 0.0  # 总的 API 请求耗时（包含传输）
        total_model_latency = 0.0  # API 返回的纯模型推理耗时
        total_fusion_time = 0.0    # insert_seg_frame（TSDF+体素化+特征）耗时
        total_merge_time = 0.0     # update_masks（掩码合并）耗时
        n_fusion = 0
        n_merge = 0
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
                    panoptic_map, segments_info, segments_count,latency_sec = sam3_api("scannet", scene_name, color_mask_img_re)
                    api_duration = time.time() - api_start
                    total_api_call_time += api_duration
                    total_model_latency += latency_sec
                    api_calls += 1
                    target_device = scene_rep.device
                    seg_image = torch.from_numpy(panoptic_map).to(device=target_device)

                    if segments_info is not None and segments_count > 0:
                        if args.feature_source == "server":
                            for i in range(segments_count):
                                mask_id = segments_info[i]['instance_id']
                                mask_class_emd = segments_info[i]['clip_embedding']
                                mask_text = segments_info[i]['category_name']
                                if mask_id is None or mask_class_emd is None:
                                    continue
                                mask_ids.append(mask_id)
                                mask_features.append(torch.as_tensor(mask_class_emd, dtype=torch.float32, device=target_device))
                                mask_texts.append(mask_text)
                            masks_features = torch.stack(mask_features, dim=0)
                        else:
                            # text / crop: compute per-mask features locally (1024-dim)
                            segs = []
                            for i in range(segments_count):
                                mask_id = segments_info[i]['instance_id']
                                # skip background (id 0) and empty masks, like server mode skips clip_embedding=None
                                if mask_id is None or mask_id == 0:
                                    continue
                                if not (panoptic_map == mask_id).any():
                                    continue
                                segs.append((mask_id, segments_info[i]['category_name']))
                            if args.feature_source == "text":
                                feats = clip_fe.text_embedding([t for _, t in segs])
                            else:  # crop
                                masks_2d = [(panoptic_map == mid) for mid, _ in segs]
                                feats = clip_fe.mask_crop_embedding(color_mask_img_re, masks_2d)
                            for (mask_id, mask_text), feat in zip(segs, feats):
                                mask_ids.append(mask_id)
                                mask_features.append(feat.to(device=target_device))
                                mask_texts.append(mask_text)
                            masks_features = torch.stack(mask_features, dim=0)

                    else:
                        seg_flag = True
                        continue

                    if (frame_id % cfg_oas["seg"]["seg_add_interval"] == 0 or seg_flag) and len(mask_features) != 0:
                        t_fusion0 = time.time()
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
                        total_fusion_time += time.time() - t_fusion0
                        n_fusion += 1
                        seg_flag = False
                        if frame_id > 0 and frame_id % scene_rep.merge_frame_interval == 0:
                            t_merge0 = time.time()
                            scene_rep.update_masks(frame_id)
                            total_merge_time += time.time() - t_merge0
                            n_merge += 1

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
        # ---- E5 per-stage breakdown (E5: per-stage + keyframe timing table) ----
        other_time = max(adjusted_elapsed - total_model_latency - total_fusion_time - total_merge_time, 0.0)
        # per-stage throughput (fps): calls per second of each stage's busy time
        stage = {
            "sam3_fps": api_calls / max(total_model_latency, 1e-6),
            "fusion_fps": n_fusion / max(total_fusion_time, 1e-6),
            "merge_fps": n_merge / max(total_merge_time, 1e-6),
            "sam3_ms": total_model_latency / max(api_calls, 1) * 1000,
            "fusion_ms": total_fusion_time / max(n_fusion, 1) * 1000,
            "merge_ms": total_merge_time / max(n_merge, 1) * 1000,
        }
        fps_msg = (
            f"Scene {scene_name} finished successfully. FPS (exclude API transfer): {fps:.2f} "
            f"({processed_frames} frames in {adjusted_elapsed:.2f}s, removed {transfer_overhead:.2f}s transfer over {api_calls} calls) | "
            f"E5 per-stage: SAM3-model {total_model_latency:.2f}s/{api_calls} calls "
            f"(avg {stage['sam3_ms']:.0f} ms, {stage['sam3_fps']:.2f} call/s), "
            f"fusion(insert_seg_frame) {total_fusion_time:.2f}s/{n_fusion} "
            f"(avg {stage['fusion_ms']:.0f} ms, {stage['fusion_fps']:.2f} frame/s), "
            f"merge(update_masks) {total_merge_time:.2f}s/{n_merge} "
            f"(avg {stage['merge_ms']:.0f} ms, {stage['merge_fps']:.2f} merge/s), "
            f"other {other_time:.2f}s"
        )
        print(fps_msg)
        if logger:
            logger.info(fps_msg)
        scene_fps.append((scene_name, fps, stage))
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
        for item in scene_fps:
            name = item[0]
            fps = item[1]
            stage = item[2] if len(item) > 2 else None
            f.write(f"{name}: {fps:.4f} FPS")
            if stage:
                f.write(f"  | sam3 {stage['sam3_fps']:.2f} call/s "
                        f"| fusion {stage['fusion_fps']:.2f} frame/s "
                        f"| merge {stage['merge_fps']:.2f} merge/s")
            f.write("\n")
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


def has_final_ckpt(scene_output_dir):
    """该场景是否已完成：存在 final.ply 或 ckpt_final.npz 即视为完成（跳过，防重复计算）。"""
    if not os.path.isdir(scene_output_dir):
        return False
    if os.path.isfile(os.path.join(scene_output_dir, "final.ply")):
        return True
    if os.path.isfile(os.path.join(scene_output_dir, "ckpt_final.npz")):
        return True
    for name in os.listdir(scene_output_dir):
        if "ckpt_final" in name:
            return True
    return False


def main():
    parser = get_parser()
    args = parser.parse_args()

    # ---- resolve 2x2 ablation arm (A/B) to feature source, config, output dir ----
    if args.arm:
        if args.arm in ("C", "D"):
            raise SystemExit(
                "arm C/D need CropFormer masks from the CropFormer pipeline "
                "(scripts/mask_predict/mask_predict_single_seq_w_semantic.py + main.py), "
                "not main_eval_scannet.py (which uses SAM3 masks)."
            )
        args.feature_source = "text" if args.arm == "A" else "crop"
        if args.config is None:
            args.config = "./config/scannet_test_1024.yaml"
        if args.output_dir is None:
            args.output_dir = os.path.join("output_ral", args.arm)
        if args.failed_log is None:
            args.failed_log = os.path.join(args.output_dir, "failed_scenes.log")
    if args.config is None:
        args.config = "./config/scannet_test.yaml"
    if args.output_dir is None:
        args.output_dir = "./output_test/nr3d"
    if args.failed_log is None:
        args.failed_log = os.path.join(args.output_dir, "failed_scenes.log")

    mp.set_start_method("spawn", force=True)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_simple_logger(args.output_dir)
    logger.info("Arguments: " + str(args))

    merge_device = f"cuda:{args.merge_gpu}"

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
        scene_output_dir = os.path.join(args.output_dir, scene_name)
        if has_final_ckpt(scene_output_dir):
            msg = f"[Skip] {scene_name}: 已完成（存在 final.ply/ckpt_final）-> {scene_output_dir}"
            print(msg)
            if logger:
                logger.info(msg)
            failed_scenes.append((scene_name, "already done (final.ply/ckpt_final)"))
            continue
        run_single_scene(scene_dir, args, merge_device, failed_scenes, scene_fps, logger)

    if scene_fps:
        avg_fps = sum(item[1] for item in scene_fps) / len(scene_fps)
        print("\n=== 场景帧率汇总（已去除 API 传输时间） ===")
        for item in scene_fps:
            name = item[0]
            fps = item[1]
            stage = item[2] if len(item) > 2 else None
            line = f"{name}: {fps:.2f} FPS"
            if stage:
                line += (f"  [sam3 {stage['sam3_fps']:.2f} call/s, "
                         f"fusion {stage['fusion_fps']:.2f} frame/s, "
                         f"merge {stage['merge_fps']:.2f} merge/s]")
            print(line)
        print(f"平均帧率: {avg_fps:.2f} FPS")
        write_fps_log(scene_fps, args.output_dir, avg_fps)
        append_fps_to_eval_log(args.output_dir, avg_fps, logger)
    else:
        logger.info("No scene FPS recorded.")

    write_failed_log(failed_scenes, args.failed_log)


if __name__ == "__main__":
    main()
