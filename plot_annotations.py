#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:08:42 2024

@author: nadya
"""


import json, pdb, argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cmx
import matplotlib.colors as colors
from PIL import Image
import numpy as np
import os
import cv2

def collect_annotations_per_object(annotations):
    objects_events = {}
    
    for ann in annotations:
        with open(ann, 'r') as file:
            data = json.load(file)
            frame_n = ann.split('/')[-1].split('.')[0][1:]
            for inst in data[0]['instances']:
                trackId = inst['trackId']
                
                if len(inst['classValues']) == 0:
                    print('ERROR in frame: ', frame_n, ', for object: ', trackId)
                    continue
                
                bbox = inst['contour']['points']
                agent = inst['classValues'][0]['value']
                action = inst['classValues'][1]['value']
                landmark = inst['classValues'][2]['value']
                
                event = (bbox, agent, action, landmark)  
                if trackId in objects_events:
                    objects_events[trackId].append((frame_n, event))
                else:
                    objects_events[trackId] = [(frame_n, event)]
    return objects_events

def plot_each_agent(frames, annotations, output_dir):
    objects_events = collect_annotations_per_object(annotations)
    frames_folder_path = frames[0].split('/')[:-1]
    frames_folder_path = '/'.join(frames_folder_path)
    first_frame = cv2.imread(frames[0])
    
    for k, v in objects_events.items():
        output_video_path = "{}/{}.mp4".format(output_dir, k)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (first_frame.shape[1], first_frame.shape[0]))
        
        for frame, event in v:
            image_path = "{}/{}.jpg".format(frames_folder_path, frame)
    
            # Load the image
            image = cv2.imread(image_path)
            box, agent, action, landmark = event
            start_point = (int(box[0]['x']), int(box[0]['y']))
            end_point = (int(box[2]['x']), int(box[2]['y']))
            image = cv2.rectangle(image, start_point, end_point, (0, 0, 255), 5)
            
            label = '({}, {}, {})'.format(agent, landmark, action)
            label_position = (start_point[0], start_point[1] - 30)
            image = cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            video_writer.write(image)
        video_writer.release()


def plot_all_annotations(frames, annotations, output_dir):
    output_dir_frames = os.path.join(output_dir, 'frames')    
    if not os.path.exists(output_dir_frames):
        os.makedirs(output_dir_frames)
        
    first_frame = cv2.imread(frames[0])    
    output_video_path = "{}/annotated_video.mp4".format(output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (first_frame.shape[1], first_frame.shape[0]))
        
    for i, ann in enumerate(annotations):
        with open(ann, 'r') as file:
            data = json.load(file)
            frame = cv2.imread(frames[i])
            
            for inst in data[0]['instances']:
                trackId = inst['trackId']
                
                if len(inst['classValues']) != 3:
                    print('ERROR in frame: ', i, ', for object: ', trackId)
                    continue
                
                bbox = inst['contour']['points']
                event = [inst['classValues'][0]['value'], inst['classValues'][1]['value'], inst['classValues'][2]['value']] # agent, action, landmark
                
                start_point = (int(bbox[0]['x']), int(bbox[0]['y']))
                end_point = (int(bbox[2]['x']), int(bbox[2]['y']))
                image = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2) 
                
                for j, label in enumerate(event):
                    label_position = (start_point[0], start_point[1] - 15*(len(event)-j))
                    image = cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                
            cv2.imwrite('{}/frame_{}.jpg'.format(output_dir_frames, str(i)), image)
            video_writer.write(image)
    video_writer.release()
                


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='plot the annotations')
    p.add_argument('frames_dir', type=str, help='Frames directory')
    p.add_argument('annot_dir', type=str, help='Annotations directory')
    args = p.parse_args()
    
    frames = sorted(os.listdir(args.frames_dir))
    frames = [args.frames_dir + '/' + frame for frame in frames]
    annotations = sorted(os.listdir(args.annot_dir))
    annotations = [args.annot_dir + '/' + ann for ann in annotations if ann.endswith('.json')]
    
    output_dir = os.path.join(args.annot_dir, 'output')    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
        
    #plot_each_agent(frames, annotations, output_dir)
    plot_all_annotations(frames, annotations, output_dir)