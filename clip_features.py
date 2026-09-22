#!/usr/bin/env python3
"""
Local CLIP ViT-H-14 feature extractor for the 2x2 reconstruction ablation.

- text_embedding(labels):   label-bound text features (arm A), dim 1024, L2-normalized.
- mask_crop_embedding(rgb, masks): per-crop image features (arm B), same cropping
  pipeline as the CropFormer baseline (scripts/mask_predict/helpers.py).

Both arms share the SAME ViT-H-14 1024-dim space, making "text vs per-crop image"
the only difference when the 2D backbone (SAM3) is fixed.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import torch
import open_clip
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts", "mask_predict"))
from helpers import get_cropped_image, pad_into_square  # noqa: E402

MODEL_NAME = "ViT-H-14"
PRETRAINED = "models/open_clip_pytorch_model.bin"


class CLIPFeatureExtractor:
    def __init__(self, device="cuda:0"):
        pretrained = PRETRAINED if os.path.exists(PRETRAINED) else PRETRAINED
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=pretrained)
        self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.device = device

    @torch.no_grad()
    def text_embedding(self, labels):
        """labels: list[str] -> Tensor(K, 1024), L2-normalized."""
        tok = self.tokenizer(list(labels)).to(self.device)
        feats = self.model.encode_text(tok)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def mask_crop_embedding(self, rgb_bgr, masks):
        """masks: list of (H,W) bool arrays; rgb_bgr: BGR ndarray (H,W,3).
        Uses the exact CropFormer cropping pipeline (3 scales, expansion 0.1,
        pad-to-square) -> per-mask Tensor(K, 1024), L2-normalized.
        Empty masks and degenerate (zero-size) crops are skipped -> zero feature."""
        feats = []
        for m in masks:
            m = np.asarray(m)
            if m.ndim != 2 or not m.any():  # empty mask -> zero feature
                feats.append(torch.zeros(1024, device=self.device))
                continue
            crops = get_cropped_image(m, rgb_bgr)  # 3 scales
            imgs = []
            for c in crops:
                if c.shape[0] == 0 or c.shape[1] == 0:  # degenerate crop
                    continue
                imgs.append(self.preprocess(pad_into_square(Image.fromarray(c))))
            if not imgs:
                feats.append(torch.zeros(1024, device=self.device))
                continue
            imgs = torch.stack(imgs).to(self.device)
            f = self.model.encode_image(imgs)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.mean(dim=0))
        if not feats:
            return torch.zeros((0, 1024), device=self.device)
        return torch.stack(feats)
