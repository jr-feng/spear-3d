#!/usr/bin/env python3
"""
E9 (R2: "no real sensor noise / motion blur"): synthetic degradation injection for
ScanNet RGB-D streams.

Builds a degraded COPY of selected scenes under <out-root>/<scene>/ with the exact
same layout the online pipeline expects:
    <scene>/color/<id>.jpg        color (BGR, jpg)
    <scene>/depth/<id>.png        depth (uint16 mm, png)
    <scene>/pose/<id>.txt         camera-to-world pose (copied unchanged)
    <scene>/intrinsics/           camera intrinsics (copied unchanged)
and a GT mesh symlink <out-root>/<scene>_vh_clean_2.ply so the dataset's parent-dir
GT lookup still works. Then run the UNMODIFIED pipeline with
--scans-root <out-root> (and --subsample 200 --dense-seg if desired) to obtain a
robustness curve  AP / grounding-Acc vs degradation level.

Degradations (one per invocation). Each is a PHYSICS-STYLE approximation of a
real sensing failure, applied to the raw frames BEFORE the streaming pipeline:

  --degrade blur          : MOTION BLUR (affects RGB only).
        Simulates hand-held scan motion during the exposure time: the image is
        convolved with a straight line kernel along the 2D motion direction.
        - Direction: projected 2D translation estimated from neighbouring
          camera poses (world->camera), i.e. the actual scan motion, NOT a random
          angle. Falls back to horizontal when poses are missing/NaN.
        - Strength: --blur-len = kernel length in pixels (odd; 5 = mild, 25 =
          strong, 40 = severe). Longer = more smearing = lower 2D mask quality.

  --degrade noise         : DEPTH SENSOR NOISE (affects depth only).
        Simulates the standard RGB-D noise model where depth error grows with
        distance, plus invalid pixels:
          sigma_mm = --noise-a + --noise-b * d_m^1.5     (d_m = depth in metres)
        i.e. at 1m, sigma ~ a + b mm; at 4m, sigma ~ a + 8*b mm.
        - --noise-a : distance-independent floor (mm), default 0.5
        - --noise-b : growth coefficient; 0.005 = mild, 0.02 = strong,
                      0.04 = severe (sigma ~16cm at 4m)
        - --noise-drop : fraction of valid depth pixels zeroed (holes /
          sensor dropout); 0.03 mild, 0.10 strong, 0.20 severe
        Noise is zero-mean Gaussian; holes corrupt the TSDF surface.

  --degrade drop-uniform  : FRAME RATE REDUCTION (drops whole frames).
        Uniformly keeps --keep-frames of the trajectory, equivalent to lowering
        the sensor frame rate: 200->133 (10Hz->6.7Hz), 200->100 (10Hz->5Hz),
        200->67 (10Hz->3.3Hz). Fewer views = lower scene coverage.

  --degrade drop-random   : RANDOM FRAME LOSS (stochastic drop).
        Keeps a fraction --keep-ratio of frames at random (seed --seed),
        mimicking burst packet loss rather than a uniform frame-rate cut.

Frame-subset control:
  --subsample N   : first evenly subsample to N frames (identical formula to the
                    dataset's --subsample, so degraded runs match e5_out_200_dense
                    baseline frames 1:1). Default 200 for blur/noise.
  For drop-* modes pass --subsample 0 and use --keep-frames / --keep-ratio directly.

Only the frames actually kept are written, so pose/depth/color stay consistent
(the dataset filters valid ids by the intersection of all three).

Examples:
  # motion blur, two levels (on the same 200-frame subset as the e5 baseline)
  python scripts/degrade_frames.py --scans-root data_eval/scannet \
      --out-root data_eval_robust/blur_05 --degrade blur --blur-len 5 \
      --scenes scene0011_00,scene0432_00 --subsample 200
  python scripts/degrade_frames.py --scans-root data_eval/scannet \
      --out-root data_eval_robust/blur_15 --degrade blur --blur-len 15 \
      --scenes scene0011_00,scene0432_00 --subsample 200

  # depth noise, two levels
  python scripts/degrade_frames.py --scans-root data_eval/scannet \
      --out-root data_eval_robust/noise_w --degrade noise --noise-b 0.002 --noise-drop 0.01 \
      --scenes scene0011_00 --subsample 200
  python scripts/degrade_frames.py --scans-root data_eval/scannet \
      --out-root data_eval_robust/noise_s --degrade noise --noise-b 0.008 --noise-drop 0.05 \
      --scenes scene0011_00 --subsample 200

  # frame drop: keep 100 of the full sequence (uniform)  /  keep 50% (random)
  python scripts/degrade_frames.py --scans-root data_eval/scannet \
      --out-root data_eval_robust/drop_100 --degrade drop-uniform --keep-frames 100 \
      --scenes scene0011_00 --subsample 0
  python scripts/degrade_frames.py --scans-root data_eval/scannet \
      --out-root data_eval_robust/drop_0.5 --degrade drop-random --keep-ratio 0.5 --seed 0 \
      --scenes scene0011_00 --subsample 0

Original data is never modified.
"""
import argparse
import math
import os
import shutil

import cv2
import numpy as np


def _frame_ids(scene_dir, subdir, suffix):
    ids = []
    folder = os.path.join(scene_dir, subdir)
    if not os.path.isdir(folder):
        return ids
    for name in os.listdir(folder):
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    return sorted(ids)


def _subsample_ids(ids, n):
    if n is None or int(n) <= 0 or int(n) >= len(ids):
        return ids
    keep = np.round(np.linspace(0, len(ids) - 1, int(n))).astype(int)
    return [ids[i] for i in keep]


def _drop_uniform(ids, keep_frames):
    if keep_frames is None or keep_frames >= len(ids):
        return ids
    keep = np.round(np.linspace(0, len(ids) - 1, int(keep_frames))).astype(int)
    return [ids[i] for i in keep]


def _drop_random(ids, keep_ratio, seed):
    rng = np.random.RandomState(seed)
    n_keep = max(1, int(round(len(ids) * keep_ratio)))
    keep = sorted(rng.choice(len(ids), size=n_keep, replace=False).tolist())
    return [ids[i] for i in keep]


def _finite_pose(p):
    """ScanNet ships a few corrupt/NaN poses; they must never enter the blur estimate."""
    return p is not None and getattr(p, "shape", None) == (4, 4) and bool(np.isfinite(p).all())


def _motion_blur_direction(pose_cur, pose_next):
    """2D image-space motion direction (radians) from inter-frame camera translation.

    Returns None when either pose is missing/non-finite or the camera barely
    translates along the optical axis, so callers can fall back safely.
    """
    if not (_finite_pose(pose_cur) and _finite_pose(pose_next)):
        return None
    d_world = pose_next[:3, 3] - pose_cur[:3, 3]
    R_wc = pose_cur[:3, :3].T  # world -> camera
    d_cam = R_wc @ d_world
    if (not np.isfinite(d_cam).all()) or abs(d_cam[2]) < 1e-9:
        return None
    # approximate pixel motion with unit focal length (direction only)
    return math.atan2(d_cam[1] / d_cam[2], d_cam[0] / d_cam[2])


def _motion_blur_kernel(length, angle_rad):
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    if angle_rad is None or not math.isfinite(float(angle_rad)):
        angle_rad = 0.0  # fixed horizontal blur as a safe fallback
    half = length // 2
    kernel = np.zeros((length, length), dtype=np.float32)
    cx, cy = half, half
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    for t in range(-half, half + 1):
        x = int(round(cx + t * cos_a))
        y = int(round(cy + t * sin_a))
        kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1e-9)
    return kernel


def _blur_angle_for(frame_ids, poses, i, search=8):
    """Estimate the blur direction at frame_ids[i], skipping non-finite poses.

    Scans outward for the nearest usable neighbour pose; the kernel is a
    symmetric line so an earlier neighbour is as good as a later one.
    Falls back to a fixed direction when no finite pair is found.
    """
    cur = poses.get(frame_ids[i])
    for step in range(1, search + 1):
        for j in (i + step, i - step):
            if 0 <= j < len(frame_ids):
                angle = _motion_blur_direction(cur, poses.get(frame_ids[j]))
                if angle is not None:
                    return angle
    return None  # caller/kernel falls back to a fixed direction


def _apply_blur(bgr_img, angle, blur_len):
    kernel = _motion_blur_kernel(blur_len, angle)
    return cv2.filter2D(bgr_img, -1, kernel)


def _apply_depth_noise(depth_mm, noise_b, noise_a, noise_drop, rng):
    """depth_mm: uint16 array in millimetres -> degraded uint16 array."""
    d_m = depth_mm.astype(np.float64) / 1000.0
    valid = d_m > 0
    sigma_mm = np.zeros_like(d_m)
    sigma_mm[valid] = noise_a + noise_b * np.power(d_m[valid], 1.5)
    noise = rng.normal(0.0, 1.0, size=d_m.shape) * sigma_mm
    out = depth_mm.astype(np.float64) + noise
    out[~valid] = 0.0
    if noise_drop > 0:
        drop = rng.uniform(0.0, 1.0, size=d_m.shape) < noise_drop
        out[drop & valid] = 0.0
    out = np.clip(out, 0, 65535)
    return out.astype(np.uint16)


def _write_scene(src_scene, dst_scene, frame_ids, args, poses):
    for sub in ("color", "depth", "pose", "intrinsics"):
        os.makedirs(os.path.join(dst_scene, sub), exist_ok=True)
    # intrinsics copy (once)
    src_intr = os.path.join(src_scene, "intrinsics")
    dst_intr = os.path.join(dst_scene, "intrinsics")
    if os.path.isdir(src_intr):
        for name in os.listdir(src_intr):
            shutil.copy2(os.path.join(src_intr, name), os.path.join(dst_intr, name))

    rng = np.random.RandomState(args.seed)
    for i, fid in enumerate(frame_ids):
        src_c = os.path.join(src_scene, "color", f"{fid}.jpg")
        src_d = os.path.join(src_scene, "depth", f"{fid}.png")
        src_p = os.path.join(src_scene, "pose", f"{fid}.txt")
        dst_c = os.path.join(dst_scene, "color", f"{fid}.jpg")
        dst_d = os.path.join(dst_scene, "depth", f"{fid}.png")
        dst_p = os.path.join(dst_scene, "pose", f"{fid}.txt")

        if args.degrade == "blur" and os.path.isfile(src_c):
            img = cv2.imread(src_c)  # BGR
            if img is not None:
                angle = _blur_angle_for(frame_ids, poses, i)
                img = _apply_blur(img, angle, args.blur_len)
                cv2.imwrite(dst_c, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(src_c, dst_c)
        else:
            shutil.copy2(src_c, dst_c)

        if args.degrade == "noise" and os.path.isfile(src_d):
            depth = cv2.imread(src_d, cv2.IMREAD_UNCHANGED)
            if depth is not None and depth.dtype == np.uint16:
                depth = _apply_depth_noise(depth, args.noise_b, args.noise_a,
                                           args.noise_drop, rng)
                cv2.imwrite(dst_d, depth)
            else:
                shutil.copy2(src_d, dst_d)
        else:
            shutil.copy2(src_d, dst_d)

        shutil.copy2(src_p, dst_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-root", required=True, help="original ScanNet root (data_eval/scannet)")
    ap.add_argument("--out-root", required=True, help="degraded copy root, e.g. data_eval_robust/blur_05")
    ap.add_argument("--degrade", choices=["none", "blur", "noise", "drop-uniform", "drop-random"],
                    default="blur")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="scene dir names; if omitted, all dirs with color+depth+pose are used")
    ap.add_argument("--subsample", type=int, default=200,
                    help="evenly subsample frame set to N first (0 = keep all); matches dataset --subsample")
    ap.add_argument("--blur-len", type=int, default=5, help="motion-blur kernel length in px (odd)")
    ap.add_argument("--noise-a", type=float, default=0.5, help="depth noise floor, sigma in mm at d=0")
    ap.add_argument("--noise-b", type=float, default=0.004,
                    help="depth noise growth: sigma_mm = a + b * d_m^1.5")
    ap.add_argument("--noise-drop", type=float, default=0.01, help="fraction of depth pixels zeroed")
    ap.add_argument("--keep-frames", type=int, default=None, help="drop-uniform: keep N frames")
    ap.add_argument("--keep-ratio", type=float, default=None, help="drop-random: keep fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.degrade in ("drop-uniform",) and args.keep_frames is None:
        raise SystemExit("--degrade drop-uniform requires --keep-frames")
    if args.degrade == "drop-random" and args.keep_ratio is None:
        raise SystemExit("--degrade drop-random requires --keep-ratio")

    if args.scenes:
        scenes = []
        for item in args.scenes:
            scenes.extend(s.strip() for s in item.split(",") if s.strip())
    else:
        scenes = sorted(
            d for d in os.listdir(args.scans_root)
            if os.path.isdir(os.path.join(args.scans_root, d))
            and os.path.isdir(os.path.join(args.scans_root, d, "color"))
            and os.path.isdir(os.path.join(args.scans_root, d, "depth"))
            and os.path.isdir(os.path.join(args.scans_root, d, "pose"))
        )

    os.makedirs(args.out_root, exist_ok=True)
    for scene in scenes:
        src = os.path.join(args.scans_root, scene)
        dst = os.path.join(args.out_root, scene)
        os.makedirs(dst, exist_ok=True)

        color_ids = _frame_ids(src, "color", ".jpg")
        depth_ids = _frame_ids(src, "depth", ".png")
        pose_ids = _frame_ids(src, "pose", ".txt")
        valid = sorted(set(color_ids) & set(depth_ids) & set(pose_ids))
        if not valid:
            print(f"[skip] {scene}: no frames")
            continue

        # load poses (world) for blur direction
        poses = {}
        for fid in valid:
            p = np.loadtxt(os.path.join(src, "pose", f"{fid}.txt"))
            poses[fid] = p if (p.shape == (4, 4) and np.isfinite(p).all()) else None

        if args.degrade == "drop-uniform":
            keep = _drop_uniform(valid, args.keep_frames)
        elif args.degrade == "drop-random":
            keep = _drop_random(valid, args.keep_ratio, args.seed)
        else:
            keep = _subsample_ids(valid, args.subsample)

        _write_scene(src, dst, keep, args, poses)

        # GT mesh symlink at the OUT root parent level, so dataset.find_gt_ply works.
        # ScanNet stores <scene>/<scene>_vh_clean_2.ply; the dataset looks it up in
        # data_location/../  i.e. <out_root>/<scene>_vh_clean_2.ply
        gt_src = os.path.join(src, f"{scene}_vh_clean_2.ply")
        if not os.path.isfile(gt_src):  # fallback: some copies put the ply at root level
            gt_src = os.path.join(args.scans_root, f"{scene}_vh_clean_2.ply")
        gt_src = os.path.abspath(gt_src)  # absolute, so symlink survives any out-root
        gt_dst = os.path.join(args.out_root, f"{scene}_vh_clean_2.ply")
        if os.path.isfile(gt_src) and not os.path.exists(gt_dst):
            try:
                os.symlink(gt_src, gt_dst)
            except OSError:
                shutil.copy2(gt_src, gt_dst)

        print(f"[ok] {scene}: {len(valid)} valid -> kept {len(keep)} frames | degrade={args.degrade}")

    print(f"\ndone -> {args.out_root}")


if __name__ == "__main__":
    main()
