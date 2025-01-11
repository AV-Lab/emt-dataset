#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: murdism
"""


import json
import os

def read_json(filename):
    with open(filename,'r') as json_file:
        data = json.load(json_file)
        return data

def gmot_annotation(video_files,folder_path,gmot_folder):
    # Read all files in the folder 
    for vfile in video_files:
        files = sorted(os.listdir(os.path.join(folder_path, vfile)))
        # print("Reading files : ", files)
        gmot_labels = [] # For every video file
        for file in files: 
            file_path = os.path.join(folder_path, vfile, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                data = data[0]  # Assuming `data` is a list with one element

                for instance in data['instances']:  # Iterate directly over the list of instances


                    # Extract the object type (Agent attribute)
                    object_type = next(
                        (attr["value"] for attr in instance["classValues"] if attr["name"] == "Agent"), "Unknown"
                    )
                    
                    if object_type == "Vehicle_traffic_light" or object_type == "Other_traffic_light": # remove traffic lights
                        # print("Removing traffic lights")
                        continue
                    

                    # Extract trackId and object class (although class is not needed for GMOT format)
                    track_id = instance["trackId"]

                    # Extract bounding box points
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    # Calculate box width and height
                    box_width = bbox_right - bbox_left
                    box_height = bbox_bottom - bbox_top
                    
                    # Set confidence to 1.0 (or update this based on your data if available)
                    confidence = 1.0
                    
                    gmot_labels.append([
                        frame_id,      # Frame ID (derived from filename or counter)
                        track_id,      # Identity ID (trackId)
                        bbox_left,     # Box top-left X coordinate
                        bbox_top,      # Box top-left Y coordinate
                        box_width,     # Box width
                        box_height,    # Box height
                        confidence     # Confidence (set to 1.0 by default)
                    ]
                    )
                    # return gmot_line
                        # kitti_labels = convert_to_kitti(data,frame_id)
        save_labels_to_txt(gmot_labels, gmot_folder, vfile)

def convert_to_kitti(data,frame_id):
    kitti_labels = []
    data = data[0]  # Assuming `data` is a list with one element

    for instance in data['instances']:  # Iterate directly over the list of instances

        # Extract the track ID
        track_id = instance["trackId"]

        # Extract the object type (Agent attribute)
        object_type = next(
            (attr["value"] for attr in instance["classValues"] if attr["name"] == "Agent"), "Unknown"
        )
        
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
        
        # Rotation around vertical axis (in radians)
        rotation_y = instance.get("rotation", 0.0)  # Default to 0 if not specified
        
        # Add to KITTI tracking format
        kitti_labels.append([
            track_id, object_type, truncated, occluded, alpha,
            bbox_left, bbox_top, bbox_right, bbox_bottom,
            height, width, length, x, y, z, rotation_y
        ])
    
    return kitti_labels
    
    # return kitti_labels


def kitti_annoatation(video_files,folder_path,kitti_folder):
    # Read all files in the folder 
    for vfile in video_files:
        files = sorted(os.listdir(os.path.join(folder_path, vfile)))
        # print("Reading files : ", files)
        kitti_labels = [] # For every video file
        for file in files: 
            file_path = os.path.join(folder_path, vfile, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                data = data[0]  # Assuming `data` is a list with one element

                for instance in data['instances']:  # Iterate directly over the list of instances

                    # Extract the track ID
                    track_id = instance["trackId"]

                    # Extract the object type (Agent attribute)
                    object_type = next(
                        (attr["value"] for attr in instance["classValues"] if attr["name"] == "Agent"), "Unknown"
                    )

                    
                    if object_type == "Vehicle_traffic_light" or object_type == "Other_traffic_light" or  object_type =="Unknown" or  object_type =="AV": # remove traffic lights,unknown objects and AV 
                        # print(f"Removing {object_type}")
                        continue
                    if object_type=="Emergency vehicle":
                        # print("object_type: %s" % object_type)
                        # print("Emergency Vehicle Fix")
                        object_type = "Emergency_vehicle"
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
                    
                    # Rotation around vertical axis (in radians)
                    rotation_y = instance.get("rotation", 0.0)  # Default to 0 if not specified
                    
                    # Add to KITTI tracking format
                    kitti_labels.append([
                        frame_id,track_id, object_type, truncated, occluded, alpha,
                        bbox_left, bbox_top, bbox_right, bbox_bottom,
                        height, width, length, x, y, z, rotation_y
                    ])
                # kitti_labels = convert_to_kitti(data,frame_id)
        save_labels_to_txt(kitti_labels, kitti_folder, vfile)

def save_labels_to_txt(labels, folder_path, file_name):
    """
    Save label data to a .txt file in a specified folder.
    
    Args:
        labels (list of list): List of label data (converted format).
        folder_path (str): Path to the folder where files will be saved.
        file_name (str): Name of the file (without extension).
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Path for the file
    file_path = os.path.join(folder_path, f"{file_name}.txt")
    
    # Write labels to the file
    with open(file_path, "w") as file:
        for label in labels:
            line = " ".join(map(str, label))
            file.write(line + "\n")
    print(f"Saved: {file_path}")


def main():
    folder_path = "emt/annotations"
    kitti_folder = "emt/kitti_annotations"
    gmot_folder = "emt/gmot_annotations"

    video_files = os.listdir(folder_path)
    print("Reading video_files : ", video_files)

    # Ensure the KITTI and GMOT annotation folders exists
    os.makedirs(kitti_folder, exist_ok=True)
    os.makedirs(gmot_folder, exist_ok=True)

    # create KITTI annotations
    kitti_annoatation(video_files, folder_path,kitti_folder)
    # create GMOT annotations
    gmot_annotation(video_files, folder_path,gmot_folder)

               
    


if __name__ == '__main__':
    main()