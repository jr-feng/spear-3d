# import argparse
# import os, sys

# from SensorData import SensorData

# # params
# parser = argparse.ArgumentParser()
# # data paths
# parser.add_argument('--root_dir', required=True, help='path to sens file to read')
# parser.add_argument('--filename', required=True, help='path to sens file to read')
# parser.add_argument('--output_path', required=True, help='path to output folder')
# parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
# parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
# parser.add_argument('--export_poses', dest='export_poses', action='store_true')
# parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
# parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

# opt = parser.parse_args()
# print(opt)


# def main():
#   if not os.path.exists(opt.output_path):
#     os.makedirs(opt.output_path)
#   # load the data
#   sys.stdout.write('loading %s...' % opt.filename)
#   sd = SensorData(opt.filename)
#   sys.stdout.write('loaded!\n')
#   if opt.export_depth_images:
#     sd.export_depth_images(os.path.join(opt.output_path, 'depth'))
#   if opt.export_color_images:
#     sd.export_color_images(os.path.join(opt.output_path, 'color'))
#   if opt.export_poses:
#     sd.export_poses(os.path.join(opt.output_path, 'pose'))
#   if opt.export_intrinsics:
#     sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


# if __name__ == '__main__':
#     main()

import argparse
import os
import sys
import glob
from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--input_dir', required=True, help='path to directory containing scene folders with sens files')
parser.add_argument('--output_dir', required=True, help='path to output root folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print opt 
def process_sens_file(sens_file_path, output_scene_dir):

    try:
        print "loading " + sens_file_path + "..."  
        sd = SensorData(sens_file_path)
        print "loaded " + sens_file_path  
        
   
        if opt.export_depth_images:
            depth_dir = os.path.join(output_scene_dir, 'depth')
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)
            sd.export_depth_images(depth_dir)
        
        if opt.export_color_images:
            color_dir = os.path.join(output_scene_dir, 'color')
            if not os.path.exists(color_dir):
                os.makedirs(color_dir)
            sd.export_color_images(color_dir)
        
        if opt.export_poses:
            pose_dir = os.path.join(output_scene_dir, 'pose')
            if not os.path.exists(pose_dir):
                os.makedirs(pose_dir)
            sd.export_poses(pose_dir)
        
        if opt.export_intrinsics:
            intrinsic_dir = os.path.join(output_scene_dir, 'intrinsic')
            if not os.path.exists(intrinsic_dir):
                os.makedirs(intrinsic_dir)
            sd.export_intrinsics(intrinsic_dir)
            
        return True
    except Exception as e:
        print "Error processing " + sens_file_path + ": " + str(e)  
        return False

def main():

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    

    scene_folders = glob.glob(os.path.join(opt.input_dir, 'scene*'))
    
    if not scene_folders:
        print "No scene* folders found in " + opt.input_dir 
        return
    
    processed_count = 0
    total_count = len(scene_folders)
    
    for scene_folder in scene_folders:
        if not os.path.isdir(scene_folder):
            continue
            
  
        scene_name = os.path.basename(scene_folder)
        

        sens_file_pattern = os.path.join(scene_folder, scene_name + ".sens") 
        sens_files = glob.glob(sens_file_pattern)
        
        if not sens_files:
            print "No .sens file found for scene " + scene_name + " at " + sens_file_pattern
            continue
            
        sens_file = sens_files[0]
        

        output_scene_dir = os.path.join(opt.output_dir, scene_name)
        if not os.path.exists(output_scene_dir):
            os.makedirs(output_scene_dir)
        
        print "Processing " + scene_name + "..." 
        if process_sens_file(sens_file, output_scene_dir):
            processed_count += 1
            print "Successfully processed " + scene_name  
        else:
            print "Failed to process " + scene_name
    
    print "\nProcessing completed: " + str(processed_count) + "/" + str(total_count) + " scenes processed successfully"

if __name__ == '__main__':
    main()