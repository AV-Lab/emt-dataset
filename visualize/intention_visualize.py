import os
import cv2
import sys
import argparse
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import group_annotations_by_frame, compute_frames_idx
from intention.models.rnn_vanilla import RNNVanillaPredictor
from intention.models.rnn_autoregressive import RNNAutoregressivePredictor

def compute_intentions_count(annotations_file):
    """
    Computes the frequency of each intention label from the provided annotations file.

    Args:
        annotations_file (str): Path to the JSON file containing annotated data.

    Prints:
        A dictionary with intention labels as keys and their respective counts as values.
    """
    intentions = {}
    with open(annotations_file, 'r') as file:
        data = json.load(file)
        for _, v in data.items():
            for i in v['intention']:
                if i not in intentions:
                    intentions[i] = 0
                intentions[i] += 1
    print(intentions)

def process_frame(frame, data):
    """
    Draws bounding boxes and intention labels on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to process.
        data (list): A list of tuples containing bounding box coordinates, 
                     ground truth intention, and optionally predicted intention.

    Returns:
        numpy.ndarray: The processed frame with bounding boxes and intention labels.
    """
    for d in data:
        bbox = d[0]
        intention_gt = d[1]  

        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(frame, start_point, end_point, (0, 165, 255), 3)  

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        text_size_gt = cv2.getTextSize(intention_gt, font, font_scale, thickness)[0]
        label_x1_gt, label_y1_gt = start_point[0], start_point[1] - text_size_gt[1] - 8
        label_x2_gt, label_y2_gt = start_point[0] + text_size_gt[0] + 8, start_point[1]
        label_y1_gt = max(label_y1_gt, 0)  

        cv2.rectangle(frame, (label_x1_gt, label_y1_gt), (label_x2_gt, label_y2_gt), (0, 165, 255), -1)
        cv2.putText(frame, intention_gt, (label_x1_gt + 4, label_y2_gt - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if len(d) > 2 and d[2]:  
            intention_pred = d[2]  
            text_size_pred = cv2.getTextSize(intention_pred, font, font_scale, thickness)[0]
            label_x1_pred = label_x2_gt + 10  
            label_x2_pred = label_x1_pred + text_size_pred[0] + 8
            label_y1_pred = label_y1_gt  
            label_y2_pred = label_y2_gt

            cv2.rectangle(frame, (label_x1_pred, label_y1_pred), (label_x2_pred, label_y2_pred), (128, 0, 128), -1)
            cv2.putText(frame, intention_pred, (label_x1_pred + 4, label_y2_pred - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

def plot_intentions_from_annotations(video_path, intentions):
    """
    Displays a video with ground truth intention annotations overlaid.

    Args:
        video_path (str): Path to the video file.
        intentions (dict): Dictionary mapping frame indices to intention annotations.

    Press 'q' to exit the visualization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()
      
    frame_idx = 0
    frame_jdx = 1
    keep_frames = compute_frames_idx(cap)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break            
        
        if frame_idx in keep_frames:
            to_plot = [(d[1][-1], d[2]) for d in intentions[frame_jdx]]
            process_frame(frame, to_plot)    
            frame_jdx += 1
    
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            cv2.resizeWindow("Frame", 2560, 1440)
            key = cv2.waitKey(100) & 0xFF
                                    
            if key == ord('q'):
                print("Exiting visualization.")
                break

        frame_idx += 1           
     
    cap.release()
    cv2.destroyAllWindows()

def plot_predicted_intentions(video_path, intentions, predictor):
    """
    Displays a video with both ground truth and predicted intentions.

    Args:
        video_path (str): Path to the video file.
        intentions (dict): Dictionary mapping frame indices to intention annotations.
        predictor (RNNAutoregressivePredictor or RNNVanillaPredictor): The trained model for predicting intentions.

    Press 'q' to exit the visualization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()
      
    frame_idx = 0
    frame_jdx = 1
    keep_frames = compute_frames_idx(cap)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break            
        
        if frame_idx in keep_frames:
            predicted_intentions = []
            objects_to_plot = [d for d in intentions[frame_jdx] if len(d[4]) > 0]
            observed_trajs = [d[0] for d in objects_to_plot]
            predictions = predictor.predict(observed_trajs)
            predicted_intentions = [(d[1][-1], d[4][0], pred[0]) for d, pred in zip(objects_to_plot, predictions)]
            
            frame_jdx += 1
    
            process_frame(frame, predicted_intentions) 
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            cv2.resizeWindow("Frame", 2560, 1440)
            key = cv2.waitKey(100) & 0xFF
                                    
            if key == ord('q'):
                print("Exiting visualization.")
                break

        frame_idx += 1           
     
    cap.release()
    cv2.destroyAllWindows()  

if __name__ == '__main__':  
    p = argparse.ArgumentParser(description='Visualize annotations and predicted intentions.')
    p.add_argument('past_trajectory', type=int, help='Length of past trajectory to consider.')
    p.add_argument('future_trajectory', type=int, help='Prediction horizon length.')
    p.add_argument('video_path', type=str, help='Path to the input video file.')
    p.add_argument('annotations_file', type=str, help='Path to the annotations JSON file.')
    p.add_argument('--setting', default="gt", type=str, choices=['gt', 'trained'], 
                   help="Choose between visualizing ground truth intentions or predicted intentions.")
    p.add_argument('--checkpoint', type=str, default=None, help='Path to the trained model checkpoint.')
    p.add_argument('--device', type=str, default='cuda:1', choices=['cuda', 'cpu'], help='Device to run the model.')
    p.add_argument('--normalize', default=False, type=bool, help='Apply normalization to input data.')
    args = p.parse_args()
    
    intentions = group_annotations_by_frame(args.past_trajectory, args.future_trajectory, args.annotations_file, intention=True)
    if args.setting == "gt":
        plot_intentions_from_annotations(args.video_path, intentions)
    else:
        if args.checkpoint is None:
            print("To run evaluation, please provide a checkpoint file.")
            exit()
        predictor = RNNVanillaPredictor(args.past_trajectory, 
                                               args.future_trajectory, 
                                               args.device, 
                                               args.normalize, 
                                               args.checkpoint) 
        plot_predicted_intentions(args.video_path, intentions, predictor)
