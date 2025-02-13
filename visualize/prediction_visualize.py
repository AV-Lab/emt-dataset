import os
import cv2
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import group_annotations_by_frame, compute_frames_idx

    
def process_frame(frame, data):
    for d in data:
        bbox = d[1][-1] 
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        image = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2) 
        #label_position = (start_point[0], start_point[1] - 5*(len(intention)))
        #image = cv2.putText(image, intention, label_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)

def plot_predictions_from_annotations(video_path, predictions):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit 
      
    frame_idx = 0
    frame_jdx = 1
    keep_frames = compute_frames_idx(cap)
    
    while True:
        ret, frame = cap.read()
        if not ret: break            
        
        if frame_idx in keep_frames:
            process_frame(frame, predictions[frame_jdx])    
            frame_jdx += 1
    
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
    cv2.destroyAllWindows()

def plot_intentions():
    pass

if __name__ == '__main__':  

    p = argparse.ArgumentParser(description='plot the annotations')
    p.add_argument('past_trajectory', type=int, help='Past Trajectory')
    p.add_argument('future_trajectory', type=int, help='Prediction Horizon')
    p.add_argument('video_path', type=str, help='Video Path')
    p.add_argument('annotations_file', type=str, help='Annotations directory')
    args = p.parse_args()
    
    #compute_inetions_count(args.annotations_file)
    predictions = group_annotations_by_frame(args.past_trajectory, args.future_trajectory, args.annotations_file)
    plot_predictions_from_annotations(args.video_path, predictions)

