#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:14:49 2025

@author: murdism
"""


import json
import os


# mapping between intentions and actions 





#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ignore_objects = set(["Vehicle_traffic_light", "Other_traffic_light", "AV"])

def read_json(filename):
    with open(filename,'r') as json_file:
        data = json.load(json_file)
        return data

def save_labels_to_txt(labels, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.json")
    
    with open(file_path, "w") as file:
        json.dump(labels, file)
    print(f"Saved: {file_path}")

        
        
def generate_intention_annotations(raw_annotations, prediction_annotations_path):

    for ann in raw_annotations:
        files = sorted(os.listdir(ann))
        prediction_labels = {} 
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
                    if track_id not in prediction_labels:
                        prediction_labels[track_id] = {'object_id':last_id, 'class':object_type, 'frames':[], 'bbox':[]}
                        last_id += 1
                        
                    
                    # Extract bounding box points
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    
                    prediction_labels[track_id]['bbox'].append((bbox_left, bbox_top, bbox_right, bbox_bottom))
                    prediction_labels[track_id]['frames'].append(int(frame_id))

        objects_predictions = {pd['object_id']: {'class': pd['class'], 'frames': pd['frames'], 'bbox': pd['bbox']} for pd in prediction_labels.values() if len(pd['frames'] >= 20)}
        objects_predictions = dict(sorted(objects_predictions.items()))
        
        file_name = ann.split('/')[-1]
        save_labels_to_txt(prediction_labels, prediction_annotations_path, file_name)
        

def main():
    raw_annotations_path = "../data/raw_annotations"
    prediction_annotations_path = "../data/annotations/prediction_annotations"
    raw_annotations = [raw_annotations_path + '/' + f for f in os.listdir(raw_annotations_path)]
    generate_intention_annotations(raw_annotations, prediction_annotations_path)
    
if __name__ == '__main__':
    main()