# import base64, io, requests, numpy as np
# from PIL import Image

# img = Image.open("data_eval/scans/scene0011_00/color/0.jpg")
# buf = io.BytesIO()
# img.save(buf, format="PNG")
# payload = {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
# params = {"scene_name": "scene0011_00", "save_outputs": False, "return_map": True}

# resp = requests.post("http://10.21.15.113:8090/panoptic", json=payload, params=params)
# print(resp.status_code, resp.text)  # 若再报错可看原因
# resp.raise_for_status()
# data = resp.json()



import base64, io, requests, numpy as np
from PIL import Image
import cv2
import time

def _ensure_pil(image):
    """Accept np.ndarray (BGR/RGB) or PIL.Image; normalize to PIL.Image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:  # grayscale
            return Image.fromarray(arr)
        if arr.shape[-1] == 3:  # BGR -> RGB (cv2 output)
            return Image.fromarray(arr[:, :, ::-1])
        if arr.shape[-1] == 4:  # BGRA -> RGBA
            return Image.fromarray(arr[:, :, [2, 1, 0, 3]])
    raise TypeError(f"Unsupported image type for sam3_api: {type(image)}")

def sam3_api(dataset, scene_name, image_pil, retries=5, backoff=2.0):
    """调用远端 SAM3 服务；瞬时连接失败/超时自动重试（隧道闪断时自愈）。"""
    image_pil = _ensure_pil(image_pil)
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    payload = {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
    params = {"dataset_name": dataset, "scene_name": scene_name, "save_outputs": False, "return_map": True}
    last_exc = None
    for attempt in range(max(int(retries), 1)):
        try:
            resp = requests.post(
                "http://127.0.0.1:18090/panoptic", json=payload, params=params, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            panoptic_map = np.array(data["panoptic_map"], dtype=np.int32)
            segments_info = data["segments_info"]
            segments_count = data["segments_count"]
            latency_sec = data["processing_time_sec"]
            return panoptic_map, segments_info, segments_count, latency_sec
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt == max(int(retries), 1) - 1:
                break
            print(
                f"[sam3_api] 请求失败 ({type(exc).__name__}), {backoff}s 后重试 "
                f"({attempt + 1}/{retries}) scene={scene_name}",
                flush=True,
            )
            time.sleep(backoff)
    raise last_exc if last_exc is not None else RuntimeError("sam3_api failed")

def main():
    start = time.time()
    color_img_path = "data_eval/scans/scene0011_00/color/0.jpg"
    color_mask_img = cv2.imread(color_img_path)
    color_mask_img_re = cv2.resize(color_mask_img, (640, 480), interpolation=cv2.INTER_AREA)

    panoptic_map, segments_info,segments_count,latency_sec = sam3_api("scene0011_00", color_mask_img_re)
    end = time.time()
    print("inference:",latency_sec)
    print("all:",end-start)


if __name__ == "__main__":
    main()