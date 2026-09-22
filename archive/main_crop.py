import sys
import os
# repo root (this file lives in archive/): make tool/, Dataset/, Scene_rep importable
# regardless of the caller's cwd. Must be set BEFORE the other imports.
_current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
sys.path.append(os.path.join(_current_dir, "third_party", "FCGF"))
import argparse
import traceback
import time
from copy import deepcopy
import logging
from typing import List, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

import tool.config as config
from Dataset.dataset import get_dataset_test
from Scene_rep import Scene_rep


# repo root (this file lives in archive/); cropformer_dir etc. are resolved from the root
current_dir = _current_dir

# COCO Panoptic-133 category names
COCO_133_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other",
    "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes",
    "clouds", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other",
    "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other",
    "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff",
    "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic",
    "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand",
    "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw",
    "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick",
    "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other",
    "waterdrops", "window-blind", "window-other", "wood"
]
COCO_133_TEXT_CLASSES = "|".join(COCO_133_LABELS)

# CropFormer paths
cropformer_dir = os.path.join(current_dir, "third_party", "detectron2", "projects", "CropFormer")
cropformer_demo_dir = os.path.join(cropformer_dir, "demo_cropformer")
START_SCENE_NAME = "scene0221_01"

if cropformer_dir not in sys.path:
    sys.path.insert(0, cropformer_dir)
if cropformer_demo_dir not in sys.path:
    sys.path.insert(0, cropformer_demo_dir)

from mask2former import add_maskformer2_config  # noqa: E402
import predictor as cropformer_predictor  # noqa: E402
import helpers  # noqa: E402
import open_clip  # noqa: E402


def setup_cropformer_cfg(args, device: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.crop_config_file)
    cfg.merge_from_list(args.crop_opts)
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def load_clip_model(pretrained_path: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=pretrained_path if os.path.exists(pretrained_path) else "laion2b_s32b_b79k")
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model, preprocess, tokenizer


def prepare_mask_features(predictions, image_bgr: np.ndarray, clip_model, preprocess, clip_device: str, confidence_threshold: float, min_mask_pixel_size: int, text_labels: List[str], text_embeds: torch.Tensor, feature_mode: str = "crop"):
    # feature_mode: "crop" = per-crop CLIP image embedding (original OAS / arm C);
    #               "text" = label-bound CLIP text embedding of the predicted class
    #               (arm D: same fusion pipeline, only per-mask semantic feature
    #               source changes).
    instances = predictions.get("instances", None)
    if instances is None or not len(instances):
        return None, None, None

    pred_masks = instances.pred_masks
    pred_scores = instances.scores
    selected = pred_scores >= confidence_threshold
    if torch.count_nonzero(selected) == 0:
        return None, None, None

    selected_masks = pred_masks[selected]
    selected_scores = pred_scores[selected]

    mask_image = np.zeros(selected_masks.shape[1:], dtype=np.uint8)
    mask_features = []
    mask_texts: List[str] = []

    _, ranks = torch.sort(selected_scores, descending=True)
    mask_id = 1
    for idx in ranks:
        mask_tensor = selected_masks[idx]
        num_pixels = torch.sum(mask_tensor).item()
        if num_pixels < min_mask_pixel_size:
            continue

        mask_np = mask_tensor.detach().cpu().numpy().astype(bool)
        cropped_images = helpers.get_cropped_image(mask_np, image_bgr)
        if len(cropped_images) == 0:
            continue

        mask_image[mask_np] = mask_id

        input_images = [preprocess(helpers.pad_into_square(Image.fromarray(crop))) for crop in cropped_images]
        images = torch.stack(input_images).reshape(-1, 3, 224, 224).to(clip_device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-7)
        mean_feature = image_features.mean(dim=0)

        # predict this mask's class via text-matching on the crop embedding
        best_idx = -1
        if text_embeds is not None and text_labels:
            sims = torch.matmul(mean_feature, text_embeds.T)
            best_idx = int(torch.argmax(sims).item())

        if feature_mode == "text" and best_idx >= 0:
            # label-bound text embedding of the predicted class (arm D)
            mask_features.append(text_embeds[best_idx])
        else:
            # per-crop CLIP image embedding (arm C / original OAS)
            mask_features.append(mean_feature)
        mask_texts.append(text_labels[best_idx] if best_idx >= 0 else "")
        mask_id += 1

    if len(mask_features) == 0:
        return None, None, None

    mask_feature_tensor = torch.stack(mask_features, dim=0)
    mask_feature_tensor = mask_feature_tensor / (mask_feature_tensor.norm(dim=-1, keepdim=True) + 1e-7)
    seg_image = torch.from_numpy(mask_image)
    return seg_image, mask_feature_tensor, mask_texts


def build_text_embeddings(text_classes: str, clip_model, tokenizer, device: str):
    if text_classes is None or len(text_classes.strip()) == 0:
        return [], None

    groups = []
    label_names = []
    for part in text_classes.split("|"):
        synonyms = [p.strip() for p in part.split(",") if p.strip()]
        if not synonyms:
            continue
        groups.append(synonyms)
        label_names.append(synonyms[0])

    if len(groups) == 0:
        return [], None

    flat_texts = [s for syns in groups for s in syns]
    with torch.no_grad():
        tokenized = tokenizer(flat_texts).to(device)
        text_feats = clip_model.encode_text(tokenized)
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-7)

    embeds = []
    idx = 0
    for syns in groups:
        syn_feats = text_feats[idx:idx + len(syns)]
        mean_feat = syn_feats.mean(dim=0)
        mean_feat = mean_feat / (mean_feat.norm(dim=-1, keepdim=True) + 1e-7)
        embeds.append(mean_feat)
        idx += len(syns)

    text_embeds = torch.stack(embeds, dim=0)
    return label_names, text_embeds


def setup_fps_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("oas_fps")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("detectron2").setLevel(logging.ERROR)
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    logging.getLogger("d2").setLevel(logging.ERROR)
    return logger


def run_single_scene(scene_dir: str, base_args, merge_device: str, mask_device: str, failed_scenes: List[Tuple[str, str]], scene_fps: List[Tuple[str, float]], logger, clip_model, clip_preprocess, demo, text_labels: List[str], text_embeds: torch.Tensor):
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
        cfg_oas["mask"]["feature_dim"] = args.feature_dim

        dataset = get_dataset_test(args.dataset, cfg_oas, args.device)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        scene_rep = Scene_rep(cfg_oas, args, dataset, args.device)
        last_valid_c2w = torch.eye(4)
        seg_flag = False
        processed_frames = 0
        start_time = time.time()

        # ---- E5 per-stage timing (CropFormer/OAS backend) ----
        # Buckets align with the SAM3 line (main_eval_scannet.py):
        #   integrate : TSDF fusion + insert_seg_frame (fusion-side)
        #   seg       : CropFormer inference + per-mask feature (crop/text) ~ SAM3 stage
        #   merge     : mask merging (update_masks)
        #   other     : remainder (save ckpt etc.)
        total_integrate = 0.0
        total_seg = 0.0
        total_merge = 0.0
        n_integrate = 0
        n_seg = 0
        n_merge = 0

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
                    t0 = time.time()
                    frustum_block_coords, _ = scene_rep.integrate_frame(frame_id, color_img[0], depth_img[0], pose_w2c)
                    total_integrate += time.time() - t0
                    n_integrate += 1

                    color_mask_img = read_image(color_img_path[0])
                    if color_mask_img.shape[0] != args.dst_h or color_mask_img.shape[1] != args.dst_w:
                        color_mask_img = cv2.resize(color_mask_img, (args.dst_w, args.dst_h), interpolation=cv2.INTER_NEAREST)

                    t0 = time.time()
                    predictions = demo.run_on_image(color_mask_img)
                    seg_image, mask_features, mask_texts = prepare_mask_features(
                        predictions,
                        color_mask_img,
                        clip_model,
                        clip_preprocess,
                        mask_device,
                        args.confidence_threshold,
                        args.min_mask_pixel_size,
                        text_labels,
                        text_embeds,
                        getattr(args, "feature_mode", "crop"),
                    )
                    total_seg += time.time() - t0
                    n_seg += 1

                    if seg_image is None or mask_features is None:
                        seg_flag = True
                        continue

                    seg_image = seg_image.to(scene_rep.device)
                    mask_features = mask_features.to(scene_rep.device)

                    if (frame_id % cfg_oas["seg"]["seg_add_interval"] == 0 or seg_flag) and mask_features.shape[0] > 0:
                        t0 = time.time()
                        scene_rep.insert_seg_frame(
                            frame_id,
                            color_img[0],
                            depth_img[0],
                            pose_c2w[0],
                            frustum_block_coords,
                            seg_image,
                            mask_features,
                            mask_texts,
                        )
                        total_integrate += time.time() - t0  # fusion-side cost
                        seg_flag = False
                        if frame_id > 0 and frame_id % scene_rep.merge_frame_interval == 0:
                            t0 = time.time()
                            scene_rep.update_masks(frame_id)
                            total_merge += time.time() - t0
                            n_merge += 1

            if cfg_oas["save"]["ckpt_interval"] > 0 and frame_id > 0 and frame_id % cfg_oas["save"]["ckpt_interval"] == 0:
                ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id)
                os.makedirs(ckpt_save_dir, exist_ok=True)
                scene_rep.save_ckpt(frame_id, ckpt_save_dir)

        print("################ Begin to save final ckpt and seg mesh/pointcloud...")
        frame_id_final = dataset.__len__() - 1 if max_frame_num is None else max_frame_num - 1

        ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id_final)
        os.makedirs(ckpt_save_dir, exist_ok=True)

        final_ckpt_path = os.path.join(local_output_dir, "ckpt_final.npz")
        scene_rep.save_ckpt(frame_id_final, ckpt_save_dir, filter_flag=True, ckpt_path=final_ckpt_path)

        scene_rep.save_merging_result(frame_id_final, reextract=False)

        final_pc_path = os.path.join(local_output_dir, "final.ply")
        scene_rep.save_pc(scene_rep.points, scene_rep.colors, final_pc_path)

        end_time = time.time()
        elapsed = max(end_time - start_time, 1e-6)
        fps = processed_frames / elapsed
        fps_msg = f"Scene {scene_name} finished successfully. FPS: {fps:.2f} ({processed_frames} frames in {elapsed:.2f}s)"
        print(fps_msg)
        other = max(elapsed - total_integrate - total_seg - total_merge, 0.0)
        # per-stage throughput (fps)
        stage = {
            "integrate_fps": n_integrate / max(total_integrate, 1e-6),
            "seg_fps": n_seg / max(total_seg, 1e-6),
            "merge_fps": n_merge / max(total_merge, 1e-6),
            "integrate_ms": total_integrate / max(n_integrate, 1) * 1000,
            "seg_ms": total_seg / max(n_seg, 1) * 1000,
            "merge_ms": total_merge / max(n_merge, 1) * 1000,
        }
        stage_msg = (
            f"[E5-OAS] {scene_name}: frames={processed_frames} elapsed={elapsed:.2f}s FPS={fps:.3f} | "
            f"integrate {total_integrate:.2f}s/{n_integrate} "
            f"(avg {stage['integrate_ms']:.0f} ms, {stage['integrate_fps']:.2f} frame/s), "
            f"seg(CropFormer+feat) {total_seg:.2f}s/{n_seg} "
            f"(avg {stage['seg_ms']:.0f} ms, {stage['seg_fps']:.2f} frame/s), "
            f"merge {total_merge:.2f}s/{n_merge} "
            f"(avg {stage['merge_ms']:.0f} ms, {stage['merge_fps']:.2f} merge/s), "
            f"other {other:.2f}s"
        )
        print(stage_msg)
        if logger:
            logger.info(fps_msg)
            logger.info(stage_msg)
        scene_fps.append((scene_name, fps, stage))
    except Exception as exc:
        failed_scenes.append((scene_name, str(exc)))
        print(f"[Failed] Scene {scene_name}: {exc}")
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
        for item in scene_fps:
            name = item[0]
            fps = item[1]
            stage = item[2] if len(item) > 2 else None
            f.write(f"{name}: {fps:.4f} FPS")
            if stage:
                f.write(f"  | integrate {stage['integrate_fps']:.2f} frame/s "
                        f"| seg {stage['seg_fps']:.2f} frame/s "
                        f"| merge {stage['merge_fps']:.2f} merge/s")
            f.write("\n")
        f.write(f"Average FPS: {avg_fps:.4f}\n")
    print(f"已将 FPS 结果写入 {fps_log_path}")


def append_fps_to_eval_log(output_dir, avg_fps, logger=None):
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


def get_parser():
    parser = argparse.ArgumentParser(description="CropFormer + OnlineAnySeg streaming fusion across scenes")
    parser.add_argument("--crop-config-file", type=str, default="third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml")
    parser.add_argument("--crop-opts", help="KEY VALUE pairs for CropFormer config", default=["MODEL.WEIGHTS", "./models/Mask2Former_hornet_3x_576d0b.pth"], nargs=argparse.REMAINDER)
    parser.add_argument("--pretrained-path", type=str, default="./models/open_clip_pytorch_model.bin", help="Pretrained CLIP checkpoint for mask embeddings")
    parser.add_argument("--text-classes", type=str, default=COCO_133_TEXT_CLASSES, help="Target semantic classes, '|' separated; synonyms separated by ',' (default: COCO Panoptic 133 classes).")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--min-mask-pixel-size", type=int, default=500)
    parser.add_argument("--dst-h", type=int, default=480)
    parser.add_argument("--dst-w", type=int, default=640)
    parser.add_argument("--config", type=str, default="./config/scannet_test.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./output_test/scannet_cropformer")
    parser.add_argument("--scans-root", type=str, default="./data_eval/scans", help="Root directory containing scene folders to evaluate")
    parser.add_argument("--scenes", nargs="+", default=None, help="Optional specific scene names to run. If omitted, all scenes under scans-root are processed.")
    parser.add_argument("--mask-gpu", type=int, default=0, help="GPU id for CropFormer + CLIP inference")
    parser.add_argument("--merge-gpu", type=int, default=0, help="GPU id for OnlineAnySeg fusion")
    parser.add_argument("--failed-log", type=str, default="./output_test/scannet/failed_scenes_crop.log", help="Where to save failed scene list")
    parser.add_argument("--feature-dim", type=int, default=1024, help="Feature dimension for CLIP embeddings (set to 1024 for ViT-H-14)")
    parser.add_argument("--feature-mode", type=str, default="crop", choices=["crop", "text"],
                        help="Per-mask semantic feature source: crop = per-crop CLIP image embedding (arm C / original OAS); text = label-bound CLIP text embedding (arm D)")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn", force=True)

    logger = setup_fps_logger(args.output_dir)

    mask_device = f"cuda:{args.mask_gpu}"
    merge_device = f"cuda:{args.merge_gpu}"

    cfg_crop = setup_cropformer_cfg(args, mask_device)
    demo = cropformer_predictor.VisualizationDemo(cfg_crop)
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model(args.pretrained_path, mask_device)
    text_labels, text_embeds = build_text_embeddings(args.text_classes, clip_model, clip_tokenizer, mask_device)

    failed_scenes: List[Tuple[str, str]] = []
    if args.scenes:
        # explicit --scenes: process exactly these, ignore START_SCENE_NAME resume logic
        candidate_names = list(args.scenes)
    else:
        candidate_names = [
            d
            for d in sorted(os.listdir(args.scans_root))
            if os.path.isdir(os.path.join(args.scans_root, d))
        ]
        # resume support: start from START_SCENE_NAME when running the full set
        if START_SCENE_NAME in candidate_names:
            candidate_names = candidate_names[candidate_names.index(START_SCENE_NAME):]

    scene_paths = []
    for s in candidate_names:
        full = os.path.join(args.scans_root, s)
        if os.path.isdir(full):
            scene_paths.append(full)
        else:
            failed_scenes.append((s, "scene directory not found"))
            print(f"[Skip] {s}: directory not found")

    scene_fps: List[Tuple[str, float]] = []
    for scene_dir in tqdm(scene_paths, desc="Scenes", unit="scene"):
        scene_name = os.path.basename(os.path.normpath(scene_dir))
        # final_ply_path = os.path.join(args.output_dir, scene_name, "final.ply")
        # if os.path.isfile(final_ply_path):
        #     print(f"[Skip] {scene_name}: found existing result at {final_ply_path}")
        #     failed_scenes.append((scene_name, f"skipped: existing result at {final_ply_path}"))
        #     continue
        run_single_scene(scene_dir, args, merge_device, mask_device, failed_scenes, scene_fps, logger, clip_model, clip_preprocess, demo, text_labels, text_embeds)

    if scene_fps:
        avg_fps = sum(item[1] for item in scene_fps) / len(scene_fps)
        print("\n=== 场景帧率汇总 ===")
        for item in scene_fps:
            name = item[0]
            fps = item[1]
            stage = item[2] if len(item) > 2 else None
            line = f"{name}: {fps:.2f} FPS"
            if stage:
                line += (f"  [integrate {stage['integrate_fps']:.2f} frame/s, "
                         f"seg {stage['seg_fps']:.2f} frame/s, "
                         f"merge {stage['merge_fps']:.2f} merge/s]")
            print(line)
        print(f"平均帧率: {avg_fps:.2f} FPS")
        write_fps_log(scene_fps, args.output_dir, avg_fps)
        append_fps_to_eval_log(args.output_dir, avg_fps, logger=None)

    write_failed_log(failed_scenes, args.failed_log)


if __name__ == "__main__":
    main()
