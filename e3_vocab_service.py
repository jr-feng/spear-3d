#!/usr/bin/env python3
"""
E3 vocabulary-scale benchmark service (NEW FILE - does not touch sam3_service_1.py).

Purpose: quantify how the per-frame SAM3 fast-CUDA panoptic pipeline behaves as the
text-prompt vocabulary size V changes (R2: "per-frame full vocabulary / runtime grows
with vocabulary size"). Reports per-request inference latency and peak GPU memory.

Endpoint:
  POST /e3  json={"image_base64": "...", "text_prompts": ["wall", "chair", ...]}
  -> {"processing_time_sec": float, "peak_mem_mb": float,
      "segments_count": int, "segment_classes": [...], "text_prompts_len": int,
      "oom": bool, "error": str|null}

Memory-lean choices:
  * model loaded once at import (FP16, same as production pipeline)
  * torch.inference_mode(), no per-call text re-encode for embedding attachment
    (skips forward_text used only by attach_sam3_text_embeddings)
  * torch.cuda.reset_peak_memory_stats before each call, so peak_mem_mb is the
    incremental peak of THIS call (vocab-size effect), not the model footprint
  * torch.cuda.empty_cache() after every call

Run (in container /home/sam3):
  nohup python3 e3_vocab_service.py > /tmp/e3_svc.log 2>&1 &   # port 8091
"""
import base64
import json
import os
import tempfile
import time

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

import sam3_panoptic_fast_cuda as fast_cuda  # loads SAM3 model once (FP16)

app = FastAPI(title="E3 vocab-scale bench (read-only, production service untouched)")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class E3Req(BaseModel):
    image_base64: str
    text_prompts: list
    return_map: bool = False


@app.post("/e3")
async def e3(req: E3Req):
    tmp_path = None
    oom = False
    error = None
    try:
        prompts = [str(p) for p in req.text_prompts if str(p).strip()]
        if not prompts:
            return {"processing_time_sec": 0.0, "peak_mem_mb": 0.0,
                    "segments_count": 0, "segment_classes": [], "text_prompts_len": 0,
                    "oom": False, "error": "empty text_prompts"}
        image_bytes = base64.b64decode(req.image_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        stuff_classes = fast_cuda.resolve_stuff_classes(prompts, None)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            mem_start = torch.cuda.memory_allocated(0)  # baseline before this call

        infer_start = time.time()
        with torch.inference_mode():
            predictor = fast_cuda.predictor
            predictor.set_image(tmp_path)            # image encoding (V-independent)
            results = predictor(text=prompts)        # full inference w/ this vocab
            if not results:
                raise RuntimeError("predictor returned no results")
            result = results[0]
            masks, scores, classes = fast_cuda.extract_result_fields(result, device=DEVICE)
            class_names = list(prompts)
            classes = torch.clamp(classes, 0, max(len(class_names) - 1, 0))
            panoptic_cls, panoptic_inst = fast_cuda.sam3_panoptic_coarse(
                masks, scores, classes, class_names, stuff_classes,
                void_label=255, mask_threshold=0.5, min_mask_area=500,
                device=DEVICE,
            )
            panoptic_map, segments_info = fast_cuda.build_panoptic_map(
                panoptic_cls, panoptic_inst, class_names, void_label=255,
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.time() - infer_start
        peak_mb = ((torch.cuda.max_memory_allocated(0) - mem_start) / 1e6) if DEVICE.type == "cuda" else 0.0

        seg_classes = [str(s.get("category_name", "")) for s in segments_info]
        resp = {
            "processing_time_sec": round(float(infer_time), 4),
            "peak_mem_mb": round(float(peak_mb), 2),
            "segments_count": len(segments_info),
            "segment_classes": seg_classes,
            "text_prompts_len": len(prompts),
            "oom": False,
            "error": None,
        }
        if req.return_map:
            resp["panoptic_map"] = panoptic_map.cpu().numpy().tolist()
        return resp
    except torch.cuda.OutOfMemoryError as exc:  # noqa: BLE001
        oom = True
        error = f"CUDA OOM: {str(exc)[:300]}"
        return {"processing_time_sec": -1.0, "peak_mem_mb": -1.0,
                "segments_count": 0, "segment_classes": [], "text_prompts_len": 0,
                "oom": True, "error": error}
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        return {"processing_time_sec": -1.0, "peak_mem_mb": -1.0,
                "segments_count": 0, "segment_classes": [], "text_prompts_len": 0,
                "oom": False, "error": error}
    finally:
        if tmp_path is not None and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)
