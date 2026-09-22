#!/usr/bin/env python3
"""
OVI-MAP-aligned semantic evaluation (mIoU / mAcc) of our reconstructions.

We generate, per scene, the two meshes OVI-MAP's eval_sem_seg expects:
  - gt_semantic_mesh.ply   : GT mesh vertices + vertex label = ScanNet200 raw id (0 bg)
  - semantic_map_gt_*.ply  : same vertices + label = our predicted class raw id (0 uncovered)
and run OVI-MAP's own eval_per_class_IoU (confusion matrix over GT-present classes),
the exact protocol behind its reported ScanNet mIoU/mAcc (Table 3).

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/ovimap_aligned_eval.py \
      --result_dirs /home/OnlineAnySeg/e5_out_200_dense output_ral/A \
      --gt_dir /home/OnlineAnySeg/data_eval/scannet \
      --scenes scene0011_00,scene0432_00,scene0527_00,scene0559_00,scene0689_00 \
      --tmp /tmp/ovimap_eval
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "OVI-MAP"))
sys.path.insert(0, os.path.join(ROOT, "OVI-MAP", "scripts"))

_real_argv = list(sys.argv)
sys.argv = ["evaluate_seqs_scannet.py", "--result_dir", ".", "--gt_dir", ".", "--class-aware"]
import evaluate_seqs_scannet as ev  # noqa: E402
from eval.utils_3d import align_gt_to_recon, get_instances_in_GT_pc  # noqa: E402
from scripts.utils.semantic_const import CLASS_LABELS_200, VALID_CLASS_IDS_200  # noqa: E402
sys.argv = _real_argv

OVI_VALID = list(VALID_CLASS_IDS_200)
OVI_LABELS = list(CLASS_LABELS_200)
OVI_ID2NAME = dict(zip(OVI_VALID, OVI_LABELS))
OVI_NAME2ID = dict(zip(OVI_LABELS, OVI_VALID))


def _norm_label(name: str) -> str:
    s = " ".join(str(name or "").strip().lower().split())
    if not s:
        return ""
    if s in OVI_NAME2ID:
        return s
    if s.endswith("s") and s[:-1] in OVI_NAME2ID:
        return s[:-1]
    ALIAS = {
        "trash can": "trash can", "wastebin": "trash can", "trashcan": "trash can",
        "bin": "trash can", "kitchen cabinet": "cabinet", "bathroom cabinet": "cabinet",
        "tv": "television", "couch": "sofa", "sofa": "sofa",
        "coffee table": "table", "dining table": "table", "table": "table",
        "monitor": "monitor", "computer monitor": "monitor",
    }
    if s in ALIAS and ALIAS[s] in OVI_NAME2ID:
        return ALIAS[s]
    return ""


def _write_sem_ply(verts, labels, out_path):
    """Write a PLY with vertex x/y/z and an int 'label' property."""
    n = verts.shape[0]
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "i4")])
    arr["x"], arr["y"], arr["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    arr["label"] = labels.astype(np.int32)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(out_path)


def build_scene_meshes(npz_path: str, gt_txt: str, gt_ply: str, recon_ply: str,
                       out_dir: str, scene: str):
    """Generate gt_semantic_mesh.ply and semantic_map_gt.ply on the GT mesh vertices."""
    os.makedirs(out_dir, exist_ok=True)
    gt_pc = o3d.io.read_point_cloud(gt_ply)
    recon_pc = o3d.io.read_point_cloud(recon_ply)
    verts = np.asarray(gt_pc.points).astype(np.float64)
    V = verts.shape[0]

    # ---- GT semantic label per vertex (raw ScanNet200 id, 0 = background) ----
    gt_ids = np.loadtxt(gt_txt)  # id*1000 + inst
    gt_sem = (gt_ids // 1000).astype(np.int32)
    gt_sem[~np.isin(gt_sem, OVI_VALID)] = 0

    # ---- predicted per-vertex semantic label (map pred masks onto GT verts) ----
    pred_sem = np.zeros(V, dtype=np.int32)
    pred_info_raw = ev.read_prediction_npz(npz_path)
    inst_masks = [v["mask"] for v in pred_info_raw.values()]
    inst_masks = np.stack(inst_masks, axis=0)
    valid_recon_pts = np.any(inst_masks, axis=0)
    corr_pt_in_recon, valid_gt_pt_indices = align_gt_to_recon(
        gt_pc, recon_pc, valid_recon_pts_mask=valid_recon_pts, distance_upper_bound=0.15
    )
    pred_info = get_instances_in_GT_pc(pred_info_raw, corr_pt_in_recon,
                                       valid_gt_pt_indices=valid_gt_pt_indices)
    for key, inst in pred_info.items():
        cls_name = _norm_label(inst.get("class_name") or inst.get("label_id") or "")
        cid = OVI_NAME2ID.get(cls_name, -1)
        if cid not in OVI_VALID:
            continue
        m = np.asarray(inst["mask"], dtype=bool)
        if len(m) != V:
            continue
        pred_sem[m] = cid

    gt_mesh_f = os.path.join(out_dir, "gt_semantic_mesh.ply")
    pred_mesh_f = os.path.join(out_dir, f"semantic_map_gt_{scene}.ply")
    _write_sem_ply(verts, gt_sem, gt_mesh_f)
    _write_sem_ply(verts, pred_sem, pred_mesh_f)
    return gt_mesh_f, pred_mesh_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dirs", nargs="+", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_seg_dir", default=os.path.join(ROOT, "eval", "scannet200", "validation"))
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--tmp", default="/tmp/ovimap_eval")
    args = ap.parse_args()

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(d for d in os.listdir(args.result_dirs[0])
                        if os.path.isdir(os.path.join(args.result_dirs[0], d)))

    from scripts.eval_sem_seg import eval_per_class_IoU  # OVI-MAP's own per-class IoU

    for rd in args.result_dirs:
        cfg_gt, cfg_pred = [], []
        for seq in scenes:
            npz = os.path.join(rd, seq, "ckpt_final.npz")
            ply = os.path.join(rd, seq, "final.ply")
            gf = os.path.join(args.gt_seg_dir, seq + ".txt")
            gp = os.path.join(args.gt_dir, seq, seq + "_vh_clean_2.ply")
            if not all(os.path.isfile(x) for x in (npz, ply, gf, gp)):
                print(f"[skip] {rd}/{seq}")
                continue
            out_dir = os.path.join(args.tmp, rd.replace("/", "_"), seq)
            try:
                gtf, pf = build_scene_meshes(npz, gf, gp, ply, out_dir, seq)
            except Exception as exc:  # noqa: BLE001
                print(f"[Failed] {rd}/{seq}: {exc}")
                continue
            cfg_gt.append({"sem_mesh_f": gtf, "inst_mesh_f": gtf, "res_folder": out_dir})
            cfg_pred.append({"sem_mesh_f": pf, "inst_mesh_f": pf, "res_folder": out_dir})
        print(f"\n=== {rd}: OVI-MAP eval_per_class_IoU over {len(cfg_gt)} scenes ===")
        if cfg_gt:
            valid_ids = [0] + OVI_VALID
            labels = ["background"] + OVI_LABELS
            eval_per_class_IoU(cfg_gt, cfg_pred, valid_ids, labels)


if __name__ == "__main__":
    main()
