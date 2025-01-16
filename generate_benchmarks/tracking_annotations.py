#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:14:49 2025

@author: murdism
"""


import json
import os
import shutil

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ignore_objects = set(["Vehicle_traffic_light", "Other_traffic_light", "AV"])

def read_json(filename):
    with open(filename,'r') as json_file:
        data = json.load(json_file)
        return data

def save_labels_to_txt(labels, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.txt")
    
    with open(file_path, "w") as file:
        for label in labels:
            line = " ".join(map(str, label))
            file.write(line + "\n")
    print(f"Saved: {file_path}")


def generate_kitti_annotations(raw_annotations, tracking_annotations_path):
    kitti_folder_path = os.path.join(tracking_annotations_path, "kitti")
    if os.path.exists(kitti_folder_path): shutil.rmtree(kitti_folder_path)
    os.makedirs(kitti_folder_path)
    
    for ann in raw_annotations:
        files = sorted(os.listdir(ann))
        kitti_labels = [] 
        tracking_ids = {}
        last_id = 1
        
        for file in files: 
            file_path = os.path.join(ann, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                
                for instance in data[0]['instances']: 
                    object_type = next(
                        (attr["value"] for attr in instance["classValues"] if attr["name"] == "Agent"), "Unknown"
                    )
                    
                    if object_type =="Unknown" or  object_type in ignore_objects: continue
                    if object_type=="Emergency vehicle": object_type = "Emergency_vehicle"
                    
                    # Extract and remap trackId
                    track_id = instance["trackId"]
                    if track_id not in tracking_ids:
                        tracking_ids[track_id] = last_id
                        last_id += 1
                    track_id = tracking_ids[track_id]
                    
                    # Extract bounding box points
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    # KITTI-specific defaults (placeholders)
                    truncated = 0.0  # Truncation (between 0 and 1, 1 being highly truncated)
                    occluded = 0  # Occlusion (0 = fully visible, 1 = partially occluded, 2 = largely occluded)
                    alpha = instance.get("rotation", 0.0)  # Observation angle (in radians)
                    
                    # 3D object dimensions (height, width, length) and location (x, y, z)
                    height = 0.0
                    width = 0.0
                    length = 0.0
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    
                    # Rotation around vertical axis, set to 0, since not available in raw data
                    rotation_y = instance.get("rotation", 0.0) 
                    
                    kitti_labels.append([
                        frame_id,track_id, object_type, truncated, occluded, alpha,
                        bbox_left, bbox_top, bbox_right, bbox_bottom,
                        height, width, length, x, y, z, rotation_y
                    ])
    
        file_name = ann.split('/')[-1]
        save_labels_to_txt(kitti_labels, kitti_folder_path, file_name)
        
        
def generate_gmot_annotations(raw_annotations, tracking_annotations_path):
    gmot_folder_path = os.path.join(tracking_annotations_path, "gmot")
    if os.path.exists(gmot_folder_path): shutil.rmtree(gmot_folder_path)
    os.makedirs(gmot_folder_path)

    for ann in raw_annotations:
        files = sorted(os.listdir(ann))
        gmot_labels = [] 
        tracking_ids = {}
        last_id = 1
        
        
        for file in files: 
            file_path = os.path.join(ann, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                
                for instance in data[0]['instances']: 
                    object_type = next(
                        (attr["value"] for attr in instance["classValues"] if attr["name"] == "Agent"), "Unknown"
                    )
                    
                    if object_type =="Unknown" or  object_type in ignore_objects: continue
                    if object_type=="Emergency vehicle": object_type = "Emergency_vehicle"
                     
                    # Extract and remap trackId
                    track_id = instance["trackId"]
                    if track_id not in tracking_ids:
                        tracking_ids[track_id] = last_id
                        last_id += 1
                    track_id = tracking_ids[track_id]
                    
                    # Extract bounding box points
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    # Calculate box width and height
                    box_width = bbox_right - bbox_left
                    box_height = bbox_bottom - bbox_top
                    
                    # Set confidence to 1.0, since not available in raw format
                    confidence = 1.0
                    
                    gmot_labels.append([
                        frame_id,      # Frame ID (derived from filename or counter)
                        track_id,      # Identity ID (trackId)
                        bbox_left,     # Box top-left X coordinate
                        bbox_top,      # Box top-left Y coordinate
                        box_width,     # Box width
                        box_height,    # Box height
                        confidence     # Confidence (set to 1.0 by default)
                    ])
    
        file_name = ann.split('/')[-1]
        save_labels_to_txt(gmot_labels, gmot_folder_path, file_name)
        

def main():
    raw_annotations_path = "../data/raw_annotations"
    tracking_annotations_path = "../data/annotations/tracking_annotations"
    raw_annotations = [raw_annotations_path + '/' + f for f in os.listdir(raw_annotations_path)]
    generate_kitti_annotations(raw_annotations, tracking_annotations_path)
    generate_gmot_annotations(raw_annotations, tracking_annotations_path)

if __name__ == '__main__':
    main()