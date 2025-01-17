#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:08:42 2024

@author: nadya
"""

import os
import cv2
import sys
import argparse
import json

# Add the root directory to sys.path
ignore_objects = set(["Vehicle_traffic_light", "Other_traffic_light", "AV"])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import compute_frames_idx

def process_frame(frame, ann):
    for inst in ann[0]['instances']:    
        bbox = inst['contour']['points']
        event = [inst['classValues'][0]['value'], inst['classValues'][1]['value'], inst['classValues'][2]['value']] # agent, action, landmark
        
        start_point = (int(bbox[0]['x']), int(bbox[0]['y']))
        end_point = (int(bbox[2]['x']), int(bbox[2]['y']))
        image = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2) 
        
        for j, label in enumerate(event):
            label_position = (start_point[0], start_point[1] - 15*(len(event)-j))
            image = cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)


def plot_annotations(video_path, annotations_path, output_video_path=None):  

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit 
   
    # read annotations 
    annotations = [annotations_path + '/' + ann for ann in sorted(os.listdir(annotations_path))]
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, 10, (frame_width, frame_height))    
        
    frame_idx = 0
    ann_idx = 0
    keep_frames = compute_frames_idx(cap)
    
    while True:
        ret, frame = cap.read()
        if not ret: break            
        
        if frame_idx in keep_frames:
            with open(annotations[ann_idx], 'r') as file:
                data = json.load(file)
                process_frame(frame, data)    
                
                if writer: writer.write(frame)
                ann_idx += 1
        
                # Display frame
                cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                cv2.imshow("Frame", frame)
                cv2.resizeWindow("Frame", 2560, 1440)
                key = cv2.waitKey(100) & 0xFF
                        
                        
                if key == ord('q'):
                    print("Exiting visualization.")
                    break

        frame_idx += 1           
     
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
                


if __name__ == '__main__':    
    ############## Check thatnumber of extracted frames match annotations
    #videos_path = "../data/videos/"
    #videos = [videos_path + f for f in os.listdir(videos_path)]
    #for vp in videos:
    #    print(f"Video path: {vp}")  
    #    cap = cv2.VideoCapture(vp)
    #     tf = 0
    #     frame_idx = 0
    #     keep_frames = compute_frames_idx(cap)
    #     while True:
    #        ret, frame = cap.read()
    #        if not ret: break            
    #        if frame_idx in keep_frames:
    #            tf += 1                
    #        frame_idx += 1           
    #    print(f"Expected frames: {target_frames}, Extracted frames: {tf}")  
    ######################################################################
    
    
    
    annotations_path = "../data/raw_annotations/"
    annotations = [annotations_path + f for f in os.listdir(annotations_path)]
    
    count = 0
    events = {}
    
    for ann in annotations:
        print('**************************************************************/n')
        print(ann)
        print('**************************************************************/n')
        annotation_files = sorted([ann + '/' + f for f in os.listdir(ann)])

        for i, ann_file in enumerate(annotation_files):
            with open(ann_file, 'r') as file:
                data = json.load(file)
                for inst in data[0]['instances']:
                    trackId = inst['trackId']
                    
                    values = inst['classValues']
                    agent = values[0]['value']
                    location = values[1]['value']
                    action = values[2]['value']
    
                    if agent not in ignore_objects:
                        if (agent,action,location) not in events:
                            events[(agent,action,location)] = 0
                        events[(agent,action,location)] += 1
                        
        # Sort the dictionary by the 'location' part of the key
    sorted_events = dict(sorted(events.items(), key=lambda item: item[0][2]))
    
    # Print the sorted dictionary
    for key, value in sorted_events.items():
        print(f"{key}: {value}")
        
        

    #p = argparse.ArgumentParser(description='plot the annotations')
    #p.add_argument('video_path', type=str, help='Video Path')
    #p.add_argument('annotations_path', type=str, help='Annotations directory')
    #args = p.parse_args()
    
    #plot_annotations(args.video_path, args.annotations_path, 'output.mp4')