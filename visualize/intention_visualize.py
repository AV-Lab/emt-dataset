import os
import cv2
import sys
import argparse
import numpy as np
import json



import json

intentions = {}

if __name__ == '__main__':  

    p = argparse.ArgumentParser(description='plot the annotations')
    p.add_argument('videos_path', type=str, help='Videos Path')
    p.add_argument('annotations_path', type=str, help='Annotations directory')
    args = p.parse_args()
    
    
    annotations = [args.annotations_path + f for f in os.listdir(args.annotations_path)]
    videos = [args.videos_path + f for f in os.listdir(args.videos_path)]
    
    for ann in annotations:
        with open(ann, 'r') as file:
            data = json.load(file)
            for k,v in data.items():
                for i in v['intention']:
                    if i in intentions:
                        intentions[i] += 1
                    else:
                        intentions[i] = 1
                        
    print(intentions)