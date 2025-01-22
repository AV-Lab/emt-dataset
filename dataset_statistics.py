#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:45:06 2024

@author: nadya
"""
import json, argparse
import os
import statistics as s

################################## Tracking statistics ################################################################
    
annot_dir = "data/annotations/tracking_annotations/kitti"
annotations = os.listdir(annot_dir)

vehciles_tracklets = 0
pedestrian_tracklets = 0
cyclist_tracklets = 0
motorbike_tracklets = 0
tracklets_length = []

vehicles =set([        
    'Car', 
    'Large_vehicle', 
    'Medium_vehicle',
    'Bus',
    'Emergency_vehicle',
    'Small_motorised_vehicle'])

for ann in annotations:
    ann_file = os.path.join(annot_dir, ann) 
    tracklets = {}
    with open(ann_file, 'r') as file:
        for line in file:
            parts = line.strip().split()

            frame_id = parts[0]
            track_id = parts[1]
            obj_class = parts[2]
        
            if track_id not in tracklets:
                tracklets[track_id] = {'obj_class':obj_class, 'frames':[]}
            tracklets[track_id]['frames'].append(frame_id)   
            
    # count objects and compute mean trakcing length
    for k,v in tracklets.items():
        if v['obj_class'] in vehicles: vehciles_tracklets += 1
        elif v['obj_class'] == "Pedestrian": pedestrian_tracklets += 1
        elif v['obj_class'] == "Cyclist": cyclist_tracklets += 1
        elif v['obj_class'] == "Motorbike": motorbike_tracklets += 1
        tracklets_length.append(len(v['frames']))

print('Tracking Benchmark Statistics:')
print(f'Total number of vehciles: {vehciles_tracklets}')
print(f'Total number of pedestrians: {pedestrian_tracklets}')
print(f'Total number of motorbikes: {motorbike_tracklets}')
print(f'Total number of cyclists: {cyclist_tracklets}')
print(f'Mean tracklet length: {s.mean(tracklets_length)} \n')    
    
#######################################################################################################################


################################## Prediction statistics ################################################################

annot_dir = "data/annotations/prediction_annotations"
annotations = os.listdir(annot_dir)

vehciles_pred = 0
pedestrian_pred = 0
total = 0

vehicles =set([        
    'Car', 
    'Large_vehicle', 
    'Medium_vehicle',
    'Bus',
    'Emergency_vehicle',
    'Small_motorised_vehicle'])

for ann in annotations:
    ann_file = os.path.join(annot_dir, ann) 
    tracklets = {}
    with open(ann_file, 'r') as file:
        data = json.load(file)
        for k,v in data.items():
            if v['class'] == 'Pedestrian':
                pedestrian_pred += 1
            elif v['class'] in vehicles:
                vehciles_pred += 1
            total += 1
print('Prediction Benchmark Statistics:')
print(f'Total number of agents: {total}')
print(f'Total number of vehciles: {vehciles_pred}')
print(f'Total number of pedestrians: {pedestrian_pred} \n')

#######################################################################################################################


################################## Intention statistics ################################################################

annot_dir = "data/annotations/intention_annotations"
annotations = os.listdir(annot_dir)

total = 0

for ann in annotations:
    ann_file = os.path.join(annot_dir, ann) 
    with open(ann_file, 'r') as file:
        data = json.load(file)
        for k,v in data.items():
            intentions = v['intention']
            prev = intentions[0]
            total += 1
            for i in range(1, len(intentions)):
                cur = intentions[i]
                if cur != prev:
                    total += 1
                    prev = cur

print('Intention Benchmark Statistics:')                    
print(f'Total number of sequences: {total} \n')

#######################################################################################################################