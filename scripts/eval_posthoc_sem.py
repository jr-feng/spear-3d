#!/usr/bin/env python3
"""
E-OVI ablation: label-mediated binding (arm A) vs post-hoc VLM assignment (arm B)
on the SAME SAM3 instance masks, measured with OVI-MAP's own eval_per_class_IoU.

Answers R1 ①/②: what does "label-mediated feature binding" buy beyond frame rate,
vs the post-hoc open-vocabulary route (OVI-MAP-style)? Both arms share the same
instance masks, the same CLIP backbone, and the same metric; the ONLY variable is
how the per-instance semantic label is produced:

  * arm A (label-bound, current behaviour): label = SAM3 category name emitted in
    2D and bound to the mask before TSDF fusion.  # per-instance post-hoc queries: 0
  * arm B (post-hoc): label = argmax_cosine( per-crop CLIP feature of the instance,
    200-class text embeddings ). This mirrors OVI-MAP's post-reconstruction
    open-vocabulary semantic assignment.  # queries: 1 VLM embedding per instance

Both produce gt_semantic_mesh.ply + semantic_map_gt_*.ply and are scored by the
exact eval_per_class_IoU behind OVI-MAP's reported mIoU/mAcc, so the resulting
rows are directly comparable to OVI-MAP (subject to the same class taxonomy).

Usage:
  /root/anaconda3/envs/OASeg/bin/python scripts/eval_posthoc_sem.py \
      --arm A --pred-root output_ral/A \
      --gt_dir data_eval/scannet \
      --gt_seg_dir eval/scannet200/validation \
      --scenes "$(cat e3_scenes_30.txt | tr '\n' ',')" \
      --tmp /tmp/posthoc_A

  /root/anaconda3/envs/OASeg/bin/python scripts/eval_posthoc_sem.py \
      --arm B --pred-root output_ral/B \
      --gt_dir data_eval/scannet \
      --gt_seg_dir eval/scannet200/validation \
      --scenes "$(cat e3_scenes_30.txt | tr '\n' ',')" \
      --tmp /tmp/posthoc_B
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
OVI_NAME2ID = dict(zip(OVI_LABELS, OVI_VALID))


def _norm_label(name):
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
    n = verts.shape[0]
    arr = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "i4")])
    arr["x"], arr["y"], arr["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    arr["label"] = labels.astype(np.int32)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(out_path)


def load_arm_npz(npz_path):
    """Load arm A/B ckpt and return (pred_info, per-instance class-name list, sem-feature matrix).

    pred_info: dict key -> {'mask': bool over recon pts}; mirrors read_prediction_npz.
    cls_names: list[str] per instance (SAM3 category name / label-bound text).
    sem_feats: (N, 1024) float32 L2-normalized per-crop CLIP features (arm B) or
               text-bound features (arm A). None if absent.
    """
    z = np.load(npz_path, allow_pickle=False)
    masks = z["pred_masks"]            # (recon_pts, N) bool
    n_inst = masks.shape[1]
    classes = np.asarray(z["pred_classes"]).astype(str)
    feats = z["pred_sem_features"] if "pred_sem_features" in z else None  # (1024, N)
    pred_info = {}
    for i in range(n_inst):
        pred_info[f"inst_{i}"] = {
            "mask": masks[:, i].astype(bool),
            "class_name": classes[i],
            # keep the raw per-instance CLIP feature for the post-hoc branch
            "sem_feature": (feats[:, i].astype(np.float32) if feats is not None else None),
        }
    return pred_info, classes.tolist(), feats


def align_pred_to_gt(pred_info, gt_pc, recon_pc):
    """Map pred masks onto GT vertices (same logic as ovimap_aligned_eval)."""
    inst_masks = np.stack([v["mask"] for v in pred_info.values()], axis=0)  # (N, recon_pts)
    valid_recon = np.any(inst_masks, axis=0)
    corr, valid_gt_idx = align_gt_to_recon(gt_pc, recon_pc, valid_recon_pts_mask=valid_recon,
                                           distance_upper_bound=0.15)
    return get_instances_in_GT_pc(pred_info, corr, valid_gt_pt_indices=valid_gt_idx)


def posthoc_labels(sem_feats, text_feats, cls_names):
    """Assign ScanNet200 label per instance via argmax cosine to the 200 text embeddings.

    Args:
      sem_feats: (N, 1024) L2-normalized per-instance features (arm B crop CLIP)
      text_feats: (200, 1024) L2-normalized text embeddings of CLASS_LABELS_200
      cls_names: list[str] (unused, kept for symmetry / debugging)
    Returns:
      ids: (N,) raw ScanNet200 id or -1 (no valid match)
      stats: dict with query count etc.
    """
    if sem_feats is None:
        return None, None
    sim = sem_feats @ text_feats.T            # (N, 200)
    best = np.argmax(sim, axis=1)             # index into CLASS_LABELS_200
    ids = np.array([OVI_VALID[i] for i in best], dtype=np.int32)
    conf = sim[np.arange(len(best)), best]
    # keep only confident/unambiguous matches? No: mirror OVI-MAP argmax, keep all.
    stats = {"queries": int(len(cls_names)), "mean_max_sim": float(conf.mean())}
    return ids, stats


def build_scene_meshes_with_labels(npz_path, gt_txt, gt_ply, recon_ply, out_dir, scene,
                                   arm, text_feats):
    os.makedirs(out_dir, exist_ok=True)
    gt_pc = o3d.io.read_point_cloud(gt_ply)
    recon_pc = o3d.io.read_point_cloud(recon_ply)
    verts = np.asarray(gt_pc.points).astype(np.float64)
    V = verts.shape[0]

    gt_ids = np.loadtxt(gt_txt)
    gt_sem = (gt_ids // 1000).astype(np.int32)
    gt_sem[~np.isin(gt_sem, OVI_VALID)] = 0

    pred_info, cls_names, sem_feats = load_arm_npz(npz_path)
    pred_info_gt = align_pred_to_gt(pred_info, gt_pc, recon_pc)

    if arm == "A":
        # label-bound: class name string -> ScanNet200 id (current behaviour)
        ids, stats = [], {"queries": 0}
        for inst in pred_info_gt.values():
            cid = OVI_NAME2ID.get(_norm_label(inst.get("class_name") or ""), -1)
            ids.append(cid if cid in OVI_VALID else -1)
        ids = np.array(ids, dtype=np.int32)
    else:  # arm B: post-hoc argmax over CLIP text embeddings
        feats_gt = np.stack([v["sem_feature"] for v in pred_info_gt.values()], axis=0) \
            if sem_feats is not None else None
        if feats_gt is not None and feats_gt.shape[0] == len(cls_names):
            # normalize defensively (should already be L2 unit norm)
            fn = np.linalg.norm(feats_gt, axis=1, keepdims=True)
            feats_gt = feats_gt / np.maximum(fn, 1e-9)
            ids, stats = posthoc_labels(feats_gt, text_feats, cls_names)
        else:
            ids = np.full(len(pred_info_gt), -1, dtype=np.int32)
            stats = {"queries": 0}

    pred_sem = np.zeros(V, dtype=np.int32)
    unmatched = 0
    for (inst, cid) in zip(pred_info_gt.values(), ids):
        if cid is None or int(cid) not in OVI_VALID:
            unmatched += 1
            continue
        m = np.asarray(inst["mask"], dtype=bool)
        if len(m) != V:
            continue
        pred_sem[m] = int(cid)

    gt_mesh_f = os.path.join(out_dir, "gt_semantic_mesh.ply")
    pred_mesh_f = os.path.join(out_dir, f"semantic_map_gt_{scene}.ply")
    _write_sem_ply(verts, gt_sem, gt_mesh_f)
    _write_sem_ply(verts, pred_sem, pred_mesh_f)
    return gt_mesh_f, pred_mesh_f, stats, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True,
                    help="A = label-mediated binding (pred_classes); B = post-hoc CLIP argmax")
    ap.add_argument("--pred-root", required=True, help="dir with sceneXXX/ckpt_final.npz + final.ply")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_seg_dir", default=os.path.join(ROOT, "eval", "scannet200", "validation"))
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--tmp", default="/tmp/posthoc_eval")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    # 200-class text embeddings (shared across scenes; only needed for arm B)
    text_feats = None
    if args.arm == "B":
        from clip_features import CLIPFeatureExtractor
        dev = os.environ.get("NS_CLIP_DEVICE") or "cpu"
        clip_fe = CLIPFeatureExtractor(dev)
        text_feats = clip_fe.text_embedding(list(CLASS_LABELS_200)).cpu().numpy()  # (200, 1024)

    from scripts.eval_sem_seg import eval_per_class_IoU
    cfg_gt, cfg_pred = [], []
    tot_queries = 0
    tot_unmatched = 0
    for seq in scenes:
        npz = os.path.join(args.pred_root, seq, "ckpt_final.npz")
        ply = os.path.join(args.pred_root, seq, "final.ply")
        gf = os.path.join(args.gt_seg_dir, seq + ".txt")
        gp = os.path.join(args.gt_dir, seq, seq + "_vh_clean_2.ply")
        if not all(os.path.isfile(x) for x in (npz, ply, gf, gp)):
            print(f"[skip] {seq}")
            continue
        out_dir = os.path.join(args.tmp, args.pred_root.replace("/", "_"), seq)
        try:
            gtf, pf, stats, unmatched = build_scene_meshes_with_labels(
                npz, gf, gp, ply, out_dir, seq, args.arm, text_feats)
        except Exception as exc:  # noqa: BLE001
            print(f"[Failed] {seq}: {exc}")
            continue
        cfg_gt.append({"sem_mesh_f": gtf, "inst_mesh_f": gtf, "res_folder": out_dir})
        cfg_pred.append({"sem_mesh_f": pf, "inst_mesh_f": pf, "res_folder": out_dir})
        tot_unmatched += unmatched
        if stats:
            tot_queries += int(stats.get("queries", 0))

    print(f"\n=== arm {args.arm} ({args.pred_root}): {len(cfg_gt)} scenes | "
          f"post-hoc queries={tot_queries} | unmatched instances={tot_unmatched} ===")
    if cfg_gt:
        valid_ids = [0] + OVI_VALID
        labels = ["background"] + OVI_LABELS
        eval_per_class_IoU(cfg_gt, cfg_pred, valid_ids, labels)


if __name__ == "__main__":
    main()
