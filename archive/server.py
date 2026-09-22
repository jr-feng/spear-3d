import sys
sys.path.append("third_party/FCGF")
import argparse
import os
import torch
import cv2
max_frame_num = None
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

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

from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os

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


    return parser

app = Flask(__name__)


@app.route('/process-image', methods=['POST'])
def process_image():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # 打开上传的图片
        img = Image.open(file.stream)

        # === 在这里进行图像处理，例如：调整大小 ===
        processed_img = img.resize((300, 300))  # 调整图片大小为300x300

        # 将处理后的图片保存到字节流
        img_byte_arr = io.BytesIO()
        processed_img.save(img_byte_arr, format='PNG')  # 也可以根据原图格式调整为'JPEG'
        img_byte_arr.seek(0)  # 将指针移回文件开头

        # 构建返回信息（包含处理说明）
        message = "图片已成功调整为300x300像素。"

        response = send_file(
            img_byte_arr,
            mimetype='image/png',
            as_attachment=False  # True 会作为附件下载，False 直接显示
        )
        response.headers['X-Processing-Message'] = message
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = get_parser()
    args = parser.parse_args()


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

    
    demo = VisualizationDemo(cfg_mask, 
                             predefined_classes=predefined_classes, 
                             user_classes=user_classes,
                             confidence_threshold=args.confidence_threshold)
    

    
    predictions, _ = demo.run_on_image(color_mask_img_re)
    seg_image = predictions["panoptic_seg"][0]


    mask_ids=[]
    mask_features=[]
    mask_texts = []
    color_img_path = color_img_path[0]
    color_mask_img = read_image(color_img_path)
    color_mask_img_re = cv2.resize(color_mask_img, (640, 480), interpolation=cv2.INTER_AREA)

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
                    

                     







