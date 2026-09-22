import cv2
import numpy as np
import os

def save_depth_visualization(depth_path, save_prefix="depth_vis"):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        print("❌ 读取失败:", depth_path)
        return

    print("dtype:", depth.dtype)
    print("shape:", depth.shape)
    print("min:", np.min(depth), "max:", np.max(depth))

    # 转 float
    depth = depth.astype(np.float32)

    # mask掉无效值（0）
    valid_mask = depth > 0

    if np.sum(valid_mask) == 0:
        print("❌ 全是0，无法可视化")
        return

    d_min = depth[valid_mask].min()
    d_max = depth[valid_mask].max()

    print("valid min:", d_min, "valid max:", d_max)

    # 归一化（只对有效区域）
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[valid_mask] = (depth[valid_mask] - d_min) / (d_max - d_min + 1e-8)

    # 转 8-bit
    depth_vis = (depth_norm * 255).astype(np.uint8)

    # 上色
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # 保存
    gray_path = save_prefix + "_gray.png"
    color_path = save_prefix + "_color.png"

    cv2.imwrite(gray_path, depth_vis)
    cv2.imwrite(color_path, depth_color)

    print("✅ 已保存:")
    print(" -", gray_path)
    print(" -", color_path)


if __name__ == "__main__":
    save_depth_visualization("/workspace/nr3d/scans/scene0114_00/depth/380.png")  # 改成你的路径