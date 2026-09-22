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
from scene_update import SceneUpdater, load_point_cloud
import open3d as o3d
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
        default=0.18,
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
    parser.add_argument("-d", "--dataset", type=str, default="./data/scans_OAS/scene0011_00")
    # parser.add_argument("-d", "--dataset", type=str, default="./data_test/scene0011_00_split_B")
    parser.add_argument("--seq_name", default="scene0011_00")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output_dir", default="./output_test/scannet")
    # parser.add_argument("-o", "--output_dir", default="./data_test/update_output")
    # parser.add_argument("-l", "--local_scene_files", type=str, default="./data_test/update_output/update")
    # parser.add_argument("-u", "--updated_scene_output", type=str, default="./data_test/update_output/update")
        # 支持多文件：使用 nargs='+' 直接在命令行输入多个路径，或传入单个逗号分隔字符串（下方会解析）
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
        default="./data_test/update_output/update.ply",
        help="Output file path or directory. If a directory is provided, per-input outputs will be created inside it with .ply suffix."
    )


    return parser


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
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

    

    dataset = get_dataset_test(args.dataset, 
                               cfg_oas, 
                               args.device, 
                               )
    
    demo = VisualizationDemo(cfg_mask, 
                             predefined_classes=predefined_classes, 
                             user_classes=user_classes,
                             confidence_threshold=args.confidence_threshold)
    

    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    last_seg_frame_id = dataset.last_seg_frame_id
    print(last_seg_frame_id)

    scene_rep = Scene_rep(cfg_oas, args, dataset, args.device)
    last_valid_c2w = torch.eye(4)

    local_output_dir = str( os.path.join(args.output_dir, args.seq_name) )
    os.makedirs(local_output_dir, exist_ok=True)

    max_frame_num = dataset.last_seg_frame_id

    ########################################### main process ###########################################
    for frame_id, (color_img, color_img_path, depth_img, pose_c2w) in tqdm(enumerate(dataloader)):
        if max_frame_num is not None and frame_id >= max_frame_num:
            break

        if torch.isnan(pose_c2w[0]).any().item() or torch.isinf(pose_c2w[0]).any().item():
            pose_c2w[0] = last_valid_c2w
        else:
            last_valid_c2w = pose_c2w[0]

        if frame_id % cfg_oas["mapping"]["keyframe_freq"] == 0:
            with torch.no_grad():

                pose_w2c = torch.inverse(pose_c2w[0])
                frustum_block_coords, extrinsic = scene_rep.integrate_frame(frame_id, color_img[0], depth_img[0], pose_w2c)

                mask_ids=[]
                mask_features=[]
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
                        first_text = mask_text.split(',')[0].strip()
                        mask_texts.append(first_text)

                    masks_features = torch.stack(mask_features, dim = 0)
                    
                else:
                    seg_flag=True
                    continue
                     

                if  (frame_id % cfg_oas["seg"]["seg_add_interval"] == 0 or seg_flag) and len(mask_features) != 0:

                    # 2.2: add detected masks in current frame into global mask bank
                    
                    valid_mask_ids, valid_mask_voxels = scene_rep.insert_seg_frame(frame_id, color_img[0], depth_img[0], pose_c2w[0], frustum_block_coords, seg_image, mask_features,mask_texts)
                    seg_flag = False
                    if frame_id > 0 and frame_id % scene_rep.merge_frame_interval == 0:
                        scene_rep.update_masks(frame_id)

                        # inst_text, inst_id, inst_center_coords = scene_rep.LLM_object_list()
                        # qwen = LLM(inst_text, inst_id, inst_center_coords)
                        # qwen.LLM_querya()


    #     # Step 2.4: update visualized segmentation result
                    

    #     # Step 2.5: save latest ckpt
        if cfg_oas["save"]["ckpt_interval"] > 0 and frame_id > 0 and frame_id % cfg_oas["save"]["ckpt_interval"] == 0:
            if scene_rep.vis_pc_flag:
                    scene_rep.vis_pc.update()

            ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id)
            os.makedirs(ckpt_save_dir, exist_ok=True)
            scene_rep.save_ckpt(frame_id, ckpt_save_dir)


    # Step 3: save final segmentation results
    print("################ Begin to save final ckpt and seg mesh/pointcloud...")
    frame_id_final = dataset.__len__() - 1 if max_frame_num is None else max_frame_num - 1  # final frame_ID of this sequence

    ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id_final)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # save final ckpt
    final_ckpt_path = os.path.join(local_output_dir, "ckpt_final.npz")
    pred_instance_mask_list = scene_rep.save_ckpt(frame_id_final, ckpt_save_dir, filter_flag=True, ckpt_path=final_ckpt_path)

    # save final seg pc
    scene_rep.save_merging_result(frame_id_final, reextract=False)

    # save finally reconstructed pointcloud
    final_pc_path = os.path.join(local_output_dir, "final.ply")

    scene_rep.save_pc(scene_rep.points, scene_rep.colors, final_pc_path)

      ############codex
    # save finally reconstructed pointcloud

    # Build updatable global scene and optionally merge local reconstructions
    # scene_updater = SceneUpdater(
    #     cfg_oas["scene"]["voxel_size"],
    #     remove_obsolete=True,
    #     overlap_margin_voxels=0
    # )

    # seg_points = getattr(scene_rep, "seg_scene_points", None)
    # seg_colors = getattr(scene_rep, "seg_scene_colors", None)
    # if seg_points is None or seg_colors is None:
    #     seg_points = scene_rep.points
    #     seg_colors = scene_rep.colors


    # scene_updater.initialize_global_scene(seg_points, seg_colors)

    # for idx, local_scene in enumerate(args.local_scene_files):
    #     if not os.path.exists(local_scene):
    #         print(f"[SceneUpdate] Skip {local_scene}, file not found.")
    #         continue
    #     local_points, local_colors = load_point_cloud(local_scene)
    #     summary = scene_updater.update_with_local_scene(local_points, local_colors)
    #     updated_points, updated_colors = scene_updater.export_scene_arrays()
    #     save_path = args.updated_scene_output
    #     if save_path:
    #         root, ext = os.path.splitext(save_path)
    #         save_path = f"{root}_{idx}{ext}" if len(args.local_scene_files) > 1 else save_path
    #     else:
    #         base = os.path.splitext(os.path.basename(local_scene))[0]
    #         save_path = os.path.join(local_output_dir, f"updated_scene_{base}.ply")
    #     scene_rep.save_pc(updated_points, updated_colors, save_path)
    #     print(f"[SceneUpdate] Updated scene with {local_scene}: {summary} -> {save_path}")
    # ####################

    # print("Input sequence finished --------------!")

    # print("start find you what you want----------!")
    # inst_text, inst_id, inst_center_coords = scene_rep.LLM_object_list()
    # qwen = LLM(inst_text, inst_id, inst_center_coords)
    # qwen.LLM_query()
    # torch.cuda.empty_cache()








