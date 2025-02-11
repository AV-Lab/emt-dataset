#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:14:49 2025

@author: murdism
"""

import json
import os

ingore_objects = set(["Vehicle_traffic_light", "Other_traffic_light", "AV"])

def read_json(filename):
    """
    Reads a JSON file and returns its content as a dictionary.
    """
    with open(filename, 'r') as json_file:
        return json.load(json_file)

def save_labels_to_txt(labels, folder_path, file_name):
    """
    Saves the processed labels as a JSON file in the specified directory.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.json")
    with open(file_path, "w") as file:
        json.dump(labels, file)
    print(f"Saved: {file_path}")

def generate_prediction_annotations(raw_annotations, prediction_annotations_path):
    """
    Processes raw annotations, extracts relevant data, and generates structured predictions.
    Filters out non-consecutive frame sequences and sequences with insufficient length.
    """
    last_id = 1
    total_seq = 0
    
    for ann in raw_annotations:
        files = sorted(os.listdir(ann))
        prediction_labels = {}
        
        for file in files:
            file_path = os.path.join(ann, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                
                for instance in data[0]['instances']:
                    values = instance['classValues']
                    agent = values[0]['value']
                    location = values[1]['value']
                    action = values[2]['value']
                    
                    if agent == "Unknown" or agent in ingore_objects:
                        continue
                    if agent == "Emergency vehicle":
                        agent = "Emergency_vehicle"
                    if agent != "Pedestrian" and action == "Stopped" and "parking" in location:
                        continue
                    
                    track_id = instance["trackId"]
                    if (track_id, agent) not in prediction_labels:
                        prediction_labels[(track_id, agent)] = {
                            'object_id': last_id, 
                            'class': agent, 
                            'frames': [], 
                            'bbox': []
                        }
                        last_id += 1
                    
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    prediction_labels[(track_id, agent)]['bbox'].append((bbox_left, bbox_top, bbox_right, bbox_bottom))
                    prediction_labels[(track_id, agent)]['frames'].append(int(frame_id))
        
        objects_predictions = {pd['object_id']: {
            'class': pd['class'], 
            'frames': pd['frames'], 
            'bbox': pd['bbox']
        } for pd in prediction_labels.values()}
        
        objects_predictions = dict(sorted(objects_predictions.items()))
        
        split_objects_predictions = {}
        for k, v in objects_predictions.items():
            beg = 0
            first_record = True
            for idx in range(1, len(v['frames'])):
                if v['frames'][idx] - v['frames'][idx - 1] != 1:
                    if first_record:
                        key = k
                        first_record = False
                    else:
                        key = last_id
                        last_id += 1
                    split_objects_predictions[key] = {
                        'class': v['class'], 
                        'frames': v['frames'][beg:idx], 
                        'bbox': v['bbox'][beg:idx]
                    }
                    beg = idx
            
            if beg < len(v["frames"]):
                key = k if first_record else last_id
                if not first_record:
                    last_id += 1
                split_objects_predictions[key] = {
                    'class': v['class'], 
                    'frames': v['frames'][beg:], 
                    'bbox': v['bbox'][beg:]
                }
        
        filtered_objects_predictions = {k: v for k, v in split_objects_predictions.items() if len(v['frames']) >= 20}
        
        error = sum(1 for k, v in filtered_objects_predictions.items() if not (sorted(v['frames']) == list(range(min(v['frames']), max(v['frames']) + 1))))
        print(len(filtered_objects_predictions.keys()), error)
        
        file_name = ann.split('/')[-1]
        save_labels_to_txt(filtered_objects_predictions, prediction_annotations_path, file_name)
        total_seq += len(filtered_objects_predictions.keys())
    
    print(f"Total sequences: {total_seq}")

def main():
    """
    Main function to execute annotation processing and generate structured prediction data.
    """
    raw_annotations_path = "../data/raw_annotations"
    prediction_annotations_path = "../data/annotations/prediction_annotations"
    raw_annotations = [raw_annotations_path + '/' + f for f in os.listdir(raw_annotations_path)]
    generate_prediction_annotations(raw_annotations, prediction_annotations_path)

if __name__ == '__main__':
    main()
