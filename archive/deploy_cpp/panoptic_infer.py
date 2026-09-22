import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

import panoptic_postprocess_cuda as pp

current_dir = os.path.dirname(os.path.abspath(__file__))
detectron2_path = os.path.join(current_dir, "third_party", "detectron2")
if detectron2_path not in sys.path:
    sys.path.insert(0, detectron2_path)
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.visualizer import Visualizer


@dataclass
class Binding:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    is_input: bool


class TensorRTPanopticRunner:
    def __init__(self, engine_path: str, dataset: str) -> None:
        self.device = torch.device("cuda")
        torch.cuda.init()  # 预热

        # 加载引擎
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 解析 binding
        self.bindings: List[int] = [0] * self.engine.num_bindings
        self.binding_info: Dict[str, Binding] = {}
        self.input_name = None
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            shape = tuple(self.engine.get_binding_shape(idx))
            is_input = self.engine.binding_is_input(idx)
            self.binding_info[name] = Binding(name, shape, dtype, is_input)
            if is_input:
                self.input_name = name
        if self.input_name is None:
            raise RuntimeError("No input binding found")

        # 预分配输入/输出 CUDA 张量
        inp = self.binding_info[self.input_name]
        torch_dtype = torch.from_numpy(np.empty((), dtype=inp.dtype)).dtype
        self.input_shape = inp.shape
        self.input_tensor = torch.empty(size=self.input_shape, device=self.device, dtype=torch_dtype)
        self.input_host = torch.empty(size=self.input_shape, device="cpu", dtype=torch_dtype, pin_memory=True)
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            info = self.binding_info[name]
            torch_dtype = torch.from_numpy(np.empty((), dtype=info.dtype)).dtype
            t = self.input_tensor if info.is_input else torch.empty(size=info.shape, device=self.device, dtype=torch_dtype)
            self.bindings[idx] = int(t.data_ptr())
            setattr(self, f"out_{name}", t if not info.is_input else None)

        # 数据集元信息
        meta = MetadataCatalog.get(dataset)
        self.thing_ids = torch.tensor(list(meta.thing_dataset_id_to_contiguous_id.values()),
                                      dtype=torch.int64, device=self.device)
        self.label_divisor = getattr(meta, "label_divisor", 1000)
        self.void_label = getattr(meta, "ignore_label", -1)
        self.meta = meta

    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image {image_path}")
        H0, W0 = img_bgr.shape[:2]
        if len(self.input_shape) == 3:
            _, H, W = self.input_shape
        else:
            _, _, H, W = self.input_shape
        if (H0, W0) != (H, W):
            img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        chw = img_rgb.transpose(2, 0, 1)
        if len(self.input_shape) == 4:
            chw = chw[None, ...]
        np.copyto(self.input_host.numpy(), chw.astype(self.binding_info[self.input_name].dtype, copy=False))
        self.input_tensor.copy_(self.input_host, non_blocking=True)
        return self.input_tensor, img_bgr, (H0, W0)

    def infer(self, image_path: str, center_threshold: float, nms_kernel: int,
              top_k: int, stuff_area: int, output_path: str) -> None:
        inp, img_bgr, original_hw = self.preprocess(image_path)

        # 推理
        t0 = time.time()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()
        # 收集输出
        outputs: Dict[str, torch.Tensor] = {}
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if self.engine.binding_is_input(idx):
                continue
            outputs[name] = getattr(self, f"out_{name}")
        sem, center, offset = self._categorize(outputs)
        t1 = time.time()

        # 后处理
        if len(self.input_shape) == 3:
            _, h, w = self.input_shape
        else:
            _, _, h, w = self.input_shape
        resized_size = (h, w)
        height, width = original_hw
        sem_resized = sem_seg_postprocess(sem[0], resized_size, height, width)
        center_resized = sem_seg_postprocess(center[0], resized_size, height, width)
        offset_resized = sem_seg_postprocess(offset[0], resized_size, height, width)

        sem_labels = sem_resized.argmax(dim=0, keepdim=True).to(torch.int64)
        panoptic, _ = pp.get_panoptic_segmentation(
            sem_labels,
            center_resized,
            offset_resized,
            self.thing_ids,
            self.label_divisor,
            stuff_area,
            self.void_label,
            center_threshold,
            nms_kernel,
            top_k,
        )
        panoptic_cpu = panoptic.squeeze(0).to(torch.int32).cpu()
        segments_info = self._build_segments_info(panoptic_cpu, self.thing_ids.tolist(), self.label_divisor)
        vis = Visualizer(img_bgr[:, :, ::-1], metadata=self.meta, scale=1.0)
        vis_output = vis.draw_panoptic_seg_predictions(panoptic_cpu, segments_info, alpha=0.6)
        cv2.imwrite(output_path, vis_output.get_image()[:, :, ::-1])
        print(f"Inference {image_path} done. infer_time={t1 - t0:.4f}s saved={output_path}")

    @staticmethod
    def _categorize(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sem = center = offset = None
        for t in outputs.values():
            if t.ndim != 4:
                continue
            c = t.shape[1]
            if c > 10:
                sem = t
            elif c == 1:
                center = t
            elif c == 2:
                offset = t
        if sem is None or center is None or offset is None:
            raise RuntimeError("Unable to match outputs.")
        return sem, center, offset

    @staticmethod
    def _build_segments_info(panoptic: torch.Tensor, thing_ids: List[int], label_divisor: int) -> List[Dict]:
        segments = []
        pan_np = panoptic.numpy()
        thing_set = set(int(t) for t in thing_ids)
        for pan_id in np.unique(pan_np):
            if pan_id <= 0:
                continue
            cat = int(pan_id // label_divisor)
            segments.append({"id": int(pan_id), "isthing": cat in thing_set, "category_id": cat})
        return segments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="panoptic_vis.png")
    parser.add_argument("--dataset", default="coco_2017_val_panoptic")
    parser.add_argument("--center-threshold", type=float, default=0.1)
    parser.add_argument("--nms-kernel", type=int, default=41)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--stuff-area", type=int, default=4096)
    args = parser.parse_args()

    runner = TensorRTPanopticRunner(args.engine, args.dataset)
    runner.infer(
        args.image,
        args.center_threshold,
        args.nms_kernel,
        args.top_k,
        args.stuff_area,
        args.output,
    )
