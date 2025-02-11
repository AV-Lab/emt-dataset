#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:14:49 2025

@author: murdism
"""

import json
import os
import shutil

# Objects to be ignored during processing
ignore_objects = set(["Vehicle_traffic_light", "Other_traffic_light", "AV"])

def read_json(filename):
    """
    Reads a JSON file and returns the parsed data.
    
    Args:
        filename (str): Path to the JSON file.
    
    Returns:
        dict: Parsed JSON data.
    """
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        return data

def save_labels_to_txt(labels, folder_path, file_name):
    """
    Saves label data to a text file in the specified folder.
    
    Args:
        labels (list): List of label entries.
        folder_path (str): Path to the directory where the file should be saved.
        file_name (str): Name of the file (without extension).
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.txt")
    
    with open(file_path, "w") as file:
        for label in labels:
            line = " ".join(map(str, label))
            file.write(line + "\n")
    print(f"Saved: {file_path}")

def generate_kitti_annotations(raw_annotations, tracking_annotations_path):
    """
    Generates KITTI format tracking annotations from raw annotation files.
    
    Args:
        raw_annotations (list): List of paths to raw annotation directories.
        tracking_annotations_path (str): Directory to save the generated KITTI annotations.
    """
    kitti_folder_path = os.path.join(tracking_annotations_path, "kitti")
    if os.path.exists(kitti_folder_path):
        shutil.rmtree(kitti_folder_path)
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
                    
                    if object_type == "Unknown" or object_type in ignore_objects:
                        continue
                    if object_type == "Emergency vehicle":
                        object_type = "Emergency_vehicle"
                    
                    track_id = instance["trackId"]
                    if (track_id, object_type) not in tracking_ids:
                        tracking_ids[(track_id, object_type)] = last_id
                        last_id += 1
                    track_id = tracking_ids[(track_id, object_type)]
                    
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    truncated = 0.0  # Placeholder for truncation
                    occluded = 0  # Placeholder for occlusion level
                    alpha = instance.get("rotation", 0.0)  # Observation angle
                    
                    height, width, length = 0.0, 0.0, 0.0  # Placeholder values
                    x, y, z = 0.0, 0.0, 0.0  # Placeholder values
                    rotation_y = instance.get("rotation", 0.0)
                    
                    kitti_labels.append([
                        frame_id, track_id, object_type, truncated, occluded, alpha,
                        bbox_left, bbox_top, bbox_right, bbox_bottom,
                        height, width, length, x, y, z, rotation_y
                    ])
        
        file_name = ann.split('/')[-1]
        save_labels_to_txt(kitti_labels, kitti_folder_path, file_name)

def generate_gmot_annotations(raw_annotations, tracking_annotations_path):
    """
    Generates GMOT format tracking annotations from raw annotation files.
    
    Args:
        raw_annotations (list): List of paths to raw annotation directories.
        tracking_annotations_path (str): Directory to save the generated GMOT annotations.
    """
    gmot_folder_path = os.path.join(tracking_annotations_path, "gmot")
    if os.path.exists(gmot_folder_path):
        shutil.rmtree(gmot_folder_path)
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
                    
                    if object_type == "Unknown" or object_type in ignore_objects:
                        continue
                    if object_type == "Emergency vehicle":
                        object_type = "Emergency_vehicle"
                    
                    track_id = instance["trackId"]
                    if (track_id, object_type) not in tracking_ids:
                        tracking_ids[(track_id, object_type)] = last_id
                        last_id += 1
                    track_id = tracking_ids[(track_id, object_type)]
                    
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    box_width = bbox_right - bbox_left
                    box_height = bbox_bottom - bbox_top
                    confidence = 1.0  # Placeholder confidence value
                    
                    gmot_labels.append([
                        frame_id, track_id, bbox_left, bbox_top, box_width, box_height, confidence
                    ])
        
        file_name = ann.split('/')[-1]
        save_labels_to_txt(gmot_labels, gmot_folder_path, file_name)

def main():
    """
    Main function to generate KITTI and GMOT annotations from raw data.
    """
    raw_annotations_path = "../data/raw_annotations"
    tracking_annotations_path = "../data/annotations/tracking_annotations"
    raw_annotations = [raw_annotations_path + '/' + f for f in os.listdir(raw_annotations_path)]
    generate_kitti_annotations(raw_annotations, tracking_annotations_path)
    generate_gmot_annotations(raw_annotations, tracking_annotations_path)

if __name__ == '__main__':
    main()
