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
        
def generate_prediction_annotations(ann_dir, output_dir, categories):
    last_id = 1
    total_seq = 0
    
    videos = sorted(os.listdir(ann_dir), key=lambda x: int(x.split('_')[-1]))
    vids_ann = {vn : os.path.join(ann_dir, vn) for vn in videos} 
    
    for vn, path in vids_ann.items():
        files = sorted(os.listdir(path))
        prediction_labels = {} 
        
        for file in files: 
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and file.endswith(".json"):
                data = read_json(file_path)
                frame_id = file.split('.')[0]
                
                for instance in data[0]['instances']: 
                    
                    values = instance['classValues']
                    agent = values[0]['value']
                    location = values[1]['value']
                    action = values[2]['value']
                    
                    if agent =="Unknown" or agent not in categories: continue
                    if agent =="Emergency vehicle": agent = "Emergency_vehicle"
                    if agent != "Pedestrian" and action == "Stopped" and "parking" in location: continue
                     
                    # Extract and remap trackId
                    track_id = instance["trackId"]
                    if (track_id, agent) not in prediction_labels:
                        prediction_labels[(track_id, agent)] = {'object_id':last_id, 
                                                                  'class':agent, 
                                                                  'frames':[], 
                                                                  'bbox':[], 
                                                                  'intention':[]}
                        last_id += 1
                        
                    
                    # Extract bounding box points
                    points = instance["contour"]["points"]
                    bbox_left = min(point["x"] for point in points)
                    bbox_top = min(point["y"] for point in points)
                    bbox_right = max(point["x"] for point in points)
                    bbox_bottom = max(point["y"] for point in points)
                    
                    
                    prediction_labels[(track_id, agent)]['bbox'].append((bbox_left, bbox_top, bbox_right, bbox_bottom))
                    prediction_labels[(track_id, agent)]['frames'].append(int(frame_id))

        objects_predictions = {pd['object_id']: {'class': pd['class'], 
                                                   'frames': pd['frames'], 
                                                   'bbox': pd['bbox']} for pd in prediction_labels.values()}
        objects_predictions = dict(sorted(objects_predictions.items()))
        
        # We need to split to make sure that the frames sequence is consequitive
        split_objects_predictions = {}
        for k,v in objects_predictions.items():
            beg = 0
            first_record = True
            for idx in range(1, len(v['frames'])):
                if v['frames'][idx] - v['frames'][idx-1] != 1: # if not consequitive frames
                    if first_record:
                        key = k
                        first_record = False
                    else:
                        key = last_id
                        last_id += 1
                    
                    split_objects_predictions[key] =  {'class': v['class'], 'frames': v['frames'][beg:idx], 'bbox': v['bbox'][beg:idx]}
                    beg = idx
                    
         #  Final chunk for [beg : end]
            if beg < len(v["frames"]):
                if first_record:
                    key = k
                else:
                    key = last_id
                    last_id += 1
        
                split_objects_predictions[key] =  {'class': v['class'], 'frames': v['frames'][beg:], 'bbox': v['bbox'][beg:]}
                
        # filter entries with trajectories less than 20 points
        filtered_objects_predictions = {k:v for k,v in split_objects_predictions.items() if len(v['frames']) >= 20}
        
        # check if there are records with no consequitive frames
        error = 0
        for k, v in filtered_objects_predictions.items():
            if not(sorted(v['frames']) == list(range(min(v['frames']), max(v['frames']) + 1))):
                error += 1
        print(f"Number of sequences: {len(filtered_objects_predictions.keys())}, number of errors: {error}")
        
        file_name = path.split('/')[-1]
        save_labels_to_txt(filtered_objects_predictions, output_dir, file_name)
        total_seq += len(filtered_objects_predictions.keys())    
    print(f"Total sequences: {total_seq}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating prediction annotations')
    parser.add_argument('--dataset_dir', type=str, help='dataset folder path')
    args = parser.parse_args()
    
    ann_dir = os.path.join(args.dataset_dir, "raw_annotations")
    output_dir = os.path.join(args.dataset_dir, "annotations/prediction")
    os.makedirs(output_dir, exist_ok=True)
    
    categories = dict(list(AGENT_LABELS.items())[:9])        
    generate_prediction_annotations(ann_dir, output_dir, categories)
