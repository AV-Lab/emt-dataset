#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:14:49 2025

@author: murdism
"""


import json
import os
import argparse
from config.labels import AGENT_LABELS

def save_labels_to_txt(labels, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.txt")
    
    with open(file_path, "w") as file:
        for label in labels:
            line = " ".join(map(str, label))
            file.write(line + "\n")
    print(f"Saved: {file_path}")


def generate_annotations(ann_dir, output_dir, categories):
    kitti_folder_path = os.path.join(output_dir, "kitti")
    gmot_folder_path = os.path.join(output_dir, "gmot")
    os.makedirs(kitti_folder_path, exist_ok=True)
    videos = sorted(os.listdir(ann_dir), key=lambda x: int(x.split('_')[-1]))
    vids_ann = {vn : os.path.join(ann_dir, vn) for vn in videos} 

    for vn, path in vids_ann.items():
        files = sorted(os.listdir(path))
        kitti_labels = [] 
        gmot_labels = []
        tracking_ids = {}
        last_id = 1
        
        for file in files: 
            file_path = os.path.join(path, file)
            with open(file_path,'r') as json_file:
                frame_data = json.load(json_file)   
                instances = frame_data[0]["instances"]
                for inst in instances:
                    agent_type = inst["classValues"][0]["value"]
                    agent_type = agent_type.replace(" ", "_")
                    if agent_type not in categories: continue

                    frame_id = file.split('.')[0][2:]
                    
                    # Extract and remap trackId
                    track_id_str = inst["trackId"]
                    if (track_id_str, agent_type) not in tracking_ids:
                        tracking_ids[(track_id_str, agent_type)] = last_id
                        last_id += 1
                    track_id = tracking_ids[(track_id_str, agent_type)]
                    
                    # Extract bounding box points
                    points = inst["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    box_width = bbox_right - bbox_left
                    box_height = bbox_bottom - bbox_top
                    
                    # KITTI-format
                    truncated = 0.0  # Truncation (between 0 and 1, 1 being highly truncated)
                    occluded = 0  # Occlusion (0 = fully visible, 1 = partially occluded, 2 = largely occluded)
                    alpha = 0.0 # Observation angle (in radians)
                    
                    # 3D object dimensions (height, width, length) and location (x, y, z)
                    height = 0.0
                    width = 0.0
                    length = 0.0
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    rotation_y = 0.0
                                       
                    kitti_labels.append([frame_id,
                                         track_id, 
                                         agent_type, 
                                         truncated, 
                                         occluded, 
                                         alpha,
                                         bbox_left, 
                                         bbox_top, 
                                         bbox_right, 
                                         bbox_bottom,
                                         height, 
                                         width, 
                                         length, 
                                         x, y, z, 
                                         rotation_y])
    
                    # GMOT-format
                    confidence = 1.0
                    
                    gmot_labels.append([frame_id,
                                        track_id,
                                        bbox_left,
                                        bbox_top,
                                        box_width,    
                                        box_height,    
                                        confidence])
    
        save_labels_to_txt(kitti_labels, kitti_folder_path, vn)
        save_labels_to_txt(kitti_labels, gmot_folder_path, vn)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating tracking annotations')
    parser.add_argument('--dataset_dir', type=str, help='dataset folder path')
    args = parser.parse_args()
    
    ann_dir = os.path.join(args.dataset_dir, "raw_annotations")
    output_dir = os.path.join(args.dataset_dir, "annotations/tracking")
    os.makedirs(output_dir, exist_ok=True)
    
    categories = dict(list(AGENT_LABELS.items())[:9])        
    generate_annotations(ann_dir, output_dir, categories)

