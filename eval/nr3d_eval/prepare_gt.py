from tqdm import tqdm
import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval.constants import SCANNET_IDS

raw_data_dir = '/workspace/nr3d/scans'
gt_dir = 'eval/nr3d_eval/validation'
label_map_file = '/workspace/nr3d/scannetv2-labels.combined.tsv'
split_file_path = ''

CLOUD_FILE_PFIX = '_vh_clean_2'
SEGMENTS_FILE_PFIX = '.0.010000.segs.json'
AGGREGATIONS_FILE_PFIX = '_vh_clean.aggregation.json'

def export_gt(filename, label_ids, instance_ids):
    gt_data = label_ids * 1000 + instance_ids + 1
    np.savetxt(filename, gt_data, fmt='%d')

# Map the raw category id to the point cloud
def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group['segments'])
    label = group['label']

    # Map the category name to id
    label_ids = labels_pd[labels_pd['raw_category'] == label]['id']
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

    # Only store for the valid categories
    if not label_id in SCANNET_IDS:
        label_id = 0

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))

    return point_IDs[0], label_id

def handle_process(scene_path, output_path, labels_pd):
    scene_id = scene_path.split('/')[-1]
    segments_file = os.path.join(scene_path, f'{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}')
    aggregations_file = os.path.join(scene_path, f'{scene_id}{AGGREGATIONS_FILE_PFIX}')
    if not os.path.isfile(aggregations_file):
        legacy_agg = os.path.join(scene_path, f'{scene_id}.aggregation.json')
        if os.path.isfile(legacy_agg):
            aggregations_file = legacy_agg

    output_gt_file = os.path.join(output_path, f'{scene_id}.txt')

    # Load segments file
    if not os.path.isfile(segments_file) or not os.path.isfile(aggregations_file):
        return f"{scene_id}: missing segments/aggregation"
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    # Generate new labels
    labelled_pc = np.zeros((len(seg_indices), 1))
    instance_ids = np.zeros((len(seg_indices), 1))
    for group in seg_groups:
        p_inds, label_id = point_indices_from_group(seg_indices, group, labels_pd)

        labelled_pc[p_inds] = label_id
        instance_ids[p_inds] = group['id'] + 1

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    export_gt(output_gt_file, labelled_pc, instance_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=16, type=int, help='The number of parallel workers')
    parser.add_argument('--raw_data_dir', default=raw_data_dir, help='Path to ScanNet scans directory')
    parser.add_argument('--gt_dir', default=gt_dir, help='Output directory for GT txt files')
    parser.add_argument('--label_map_file', default=label_map_file, help='scannetv2-labels.combined.tsv path')
    parser.add_argument('--split_file_path', default=split_file_path, help='Optional split file (one scene id per line)')
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(config.label_map_file, sep='\t', header=0)

    if config.split_file_path:
        with open(config.split_file_path) as val_file:
            val_scenes = val_file.read().splitlines()
        scene_paths = [os.path.join(config.raw_data_dir, scene) for scene in val_scenes]
    else:
        scene_paths = [
            os.path.join(config.raw_data_dir, d)
            for d in sorted(os.listdir(config.raw_data_dir))
            if os.path.isdir(os.path.join(config.raw_data_dir, d))
        ]

    os.makedirs(config.gt_dir, exist_ok=True)

    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    print('Processing scenes...')
    # show progress
    results = list(tqdm(pool.map(handle_process, scene_paths, repeat(config.gt_dir), repeat(labels_pd)), total=len(scene_paths)))
    failed = [r for r in results if isinstance(r, str)]
    if failed:
        print(f"Skipped {len(failed)} scenes (missing files). Example: {failed[:5]}")
