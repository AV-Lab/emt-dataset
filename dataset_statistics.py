#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:45:06 2024

@author: nadya
"""
import json, argparse
import os

# Computing total number of bounding box annotations
# total number of unique agents 
# total number of vehicles
# total number of pedestrians


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='compute the dataset statistics')
    p.add_argument('annot_dir', type=str, help='Annotations directory')
    args = p.parse_args()
    
    # retireve all videos
    annotations = os.listdir(args.annot_dir)
    
    bounding_box_count = 0
    unique_agents_vehicles = set()
    unique_agents_people = set()
    unique_agents_rest = set()
    # vehicles = set(['Car', 
    #                 'Large_vehicle', 
    #                 'Medium_vehicle',
    #                 'Motorbike',
    #                 'Bus',
    #                 'Emergency_vehicle',
    #                 'Small_motorised_vehicle'])
    vehicles =set([        
        'Car', 
        'Large_vehicle', 
        'Medium_vehicle',
        'Motorbike',
        'Bus',
        'Emergency_vehicle',
        'Small_motorised_vehicle'])
    for ann in annotations:
        ann_dir = os.path.join(args.annot_dir, ann) 
        annotation_files = os.listdir(ann_dir)
        annotations_json = [ann_dir + '/' + ann_file for ann_file in annotation_files if ann_file.endswith('.json')]
    
        for ann_json in annotations_json:    
            with open(ann_json, 'r') as file:
                data = json.load(file)                
                for inst in data[0]['instances']:
                    trackId = inst['trackId']
                    
                    # if len(inst['classValues']) != 3:
                    #     continue
                    object_type = next(
                        (attr["value"] for attr in inst["classValues"] if attr["name"] == "Agent"), "Unknown"
                    )

                    if object_type == "Emergency vehicle":
                        # print("object_type: %s" % object_type)
                        # print("Emergency Vehicle Fix")
                        object_type = "Emergency_vehicle"
                    
                    if object_type == 'Pedestrian' or object_type == 'Cyclist' :
                        unique_agents_people.add(trackId)
                        bounding_box_count += 1
                    elif object_type in vehicles:
                        unique_agents_vehicles.add(trackId)
                        bounding_box_count += 1
                    else:
                        unique_agents_rest.add(trackId)
                        
    pd = len(unique_agents_people)  
    vh = len(unique_agents_vehicles)                  

    print('Total number of bounding box annotations: ', bounding_box_count)
    print('Total number of people (pedestrians and Cylicsts): ', pd)
    print('Total number of vehicles: ', vh)
    print('Total number of unique agents: ', pd + vh)
        
        
    


