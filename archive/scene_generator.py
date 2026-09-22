from tqdm import tqdm
from torch.utils.data import DataLoader
from detectron2.data.detection_utils import read_image
import torch
import cv2
import os

class Scene_generator:
    def __init__(self, dataset, scene_rep, model, cfg_oas):
        self.dataset = dataset
        self.scene_rep = scene_rep
        self.model = model
        self.last_seg_frame_id = dataset.last_seg_frame_id
        self.dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        self.cfg_oas = cfg_oas
        self.last_valid_c2w = torch.eye(4)

    def get_instance(self):
        mask_weight_threshold = min(self.scene_rep.merge_time + 2, self.cfg_oas["seg"]["mask_weight_threshold"])  # only merged_mask with weight>threshold will be visualized
        valid_merged_mask_ids = torch.where(self.scene_rep.merged_mask_weight[:self.scene_rep.c_mask_num] >= mask_weight_threshold)[0]
        valid_instance_list = [self.scene_rep.maskGraph.instance_dict[i] for i in valid_merged_mask_ids.tolist()]
        
        inst_center_coords = [instance.get_mask_center_coords for instance in valid_instance_list]
        inst_text = [instance.get_instance_text for instance in valid_instance_list]
        inst_id = [instance.get_mask_id for instance in valid_instance_list]
        inst_coords  = [instance.get_mask_voxel_coords for instance in valid_instance_list]
        inst_indices = [instance.get_mask_voxel_indices for instance in valid_instance_list]
        inst_sem_feature = [instance.get_semantic_feature for instance in valid_instance_list]
         
        return valid_instance_list,inst_text, inst_id, inst_center_coords, inst_coords, inst_indices, inst_sem_feature

    def generator(self):
            
        for frame_id, (color_img, color_img_path, depth_img, pose_c2w) in tqdm(enumerate(self.dataloader)):
            if self.last_seg_frame_id is not None and frame_id >=self.last_seg_frame_id:
                break

            if torch.isnan(pose_c2w[0]).any().item() or torch.isinf(pose_c2w[0]).any().item():
                pose_c2w[0] = self.last_valid_c2w
            else:
                self.last_valid_c2w = pose_c2w[0]

            if frame_id % self.cfg_oas["mapping"]["keyframe_freq"] == 0:
                with torch.no_grad():

                    pose_w2c = torch.inverse(pose_c2w[0])
                    frustum_block_coords, extrinsic = self.scene_rep.integrate_frame(frame_id, color_img[0], depth_img[0], pose_w2c)

                    mask_ids=[]
                    mask_features=[]
                    mask_texts = []
                    color_img_path = color_img_path[0]
                    color_mask_img = read_image(color_img_path)
                    color_mask_img_re = cv2.resize(color_mask_img, (640, 480), interpolation=cv2.INTER_AREA)

                    predictions, _ = self.model.run_on_image(color_mask_img_re)
                    seg_image = predictions["panoptic_seg"][0]
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
                        
                    else:
                        continue
                        

                    if  (frame_id % self.cfg_oas["seg"]["seg_add_interval"] == 0 ) and len(mask_features) != 0:

                        # 2.2: add detected masks in current frame into global mask bank
                        valid_mask_ids, valid_mask_voxels = self.scene_rep.insert_seg_frame(frame_id, color_img[0], depth_img[0], pose_c2w[0], frustum_block_coords, seg_image, mask_features, mask_texts)
                        print(valid_mask_ids)

                        if frame_id > 0 and frame_id % self.scene_rep.merge_frame_interval == 0:
                            self.scene_rep.update_masks(frame_id)


        inst_list, inst_text, inst_id, inst_center_coords, inst_coords, inst_indices, inst_sem_feature = self.get_instance()
        return inst_list, inst_text, inst_id, inst_center_coords, inst_coords, inst_indices, inst_sem_feature
