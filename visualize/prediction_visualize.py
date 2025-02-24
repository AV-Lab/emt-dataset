import os
import cv2
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import group_annotations_by_frame, compute_frames_idx
from prediction.models.rnn import RNNPredictor
from prediction.models.gcn import GCNPredictor
from prediction.models.gcn_temporal import GCNLSTMPredictor
from prediction.models.gat import GATPredictor
from prediction.models.gat_temporal import GATLSTMPredictor
from evaluation.distance_metrics import calculate_ade, calculate_fde
    
        
def process_frame(frame, data):
    """
    Draw the past and future trajectories for each object on the frame.
    
    Args:
        frame (np.array): The current video frame.
        data (list): List of predictions/annotations for this frame. Each element is a tuple:
                     (object_id, trajectory), where trajectory is a list of (x, y) coordinates.
    """
    for d in data:
        future_traj = d[2]
        bbox = d[1][-1] 
        
        if len(future_traj) > 0: 
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2) 
    
            for i in range(1, len(future_traj)):
                start_pt = (int(future_traj[i-1][0]), int(future_traj[i-1][1]))
                end_pt = (int(future_traj[i][0]), int(future_traj[i][1]))
                cv2.line(frame, start_pt, end_pt, (0, 0, 255), 2)
                cv2.circle(frame, start_pt, 3, (0, 0, 255), -1)
        
            if len(future_traj) >= 2:
                start_arrow = (int(future_traj[-2][0]), int(future_traj[-2][1]))
                end_arrow = (int(future_traj[-1][0]), int(future_traj[-1][1]))
                cv2.arrowedLine(frame, start_arrow, end_arrow, (0, 0, 255), 2, tipLength=0.3)
            else:
                cv2.circle(frame, (int(future_traj[-1][0]), int(future_traj[-1][1])), 5, (0, 0, 255), -1)


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
        if not ret:
            break
        
        if frame_idx in keep_frames:
            if frame_jdx < len(predictions):
                process_frame(frame, predictions[frame_jdx])
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

def plot_predictions(video_path, trajectories, predictor):
    """
    Displays a video with both ground truth and predicted intentions.

    Args:
        video_path (str): Path to the video file.
        intentions (dict): Dictionary mapping frame indices to intention annotations.
        predictor (LSTM, GNN, GAT): The trained model for predicting intentions.

    Press 'q' to exit the visualization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()
      
    frame_idx = 0
    frame_jdx = 1
    keep_frames = compute_frames_idx(cap)
    
    ade = 0
    fde = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break            
        
        if frame_idx in keep_frames:
            agents_trajectories = []
            if frame_jdx in trajectories: 
                objects_to_plot = [d for d in trajectories[frame_jdx] if len(d[2]) > 0]
                if len(objects_to_plot) > 0:
                    observed_trajs = [d[0] for d in objects_to_plot]
                    print(len(observed_trajs))
                    predictions = predictor.predict(observed_trajs)
                    print(len(predictions))
                    agents_trajectories = [(d[1][-1], d[2], pred) for d, pred in zip(objects_to_plot, predictions)]
                    
                    target_positions = [at[1] for at in agents_trajectories]
                    pred_positions = [at[2][:len(at[1])] for at in agents_trajectories]
                    
                    ade += calculate_ade(pred_positions, target_positions)
                    fde += calculate_fde(pred_positions, target_positions)
            
            frame_jdx += 1
            #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            #cv2.imshow("Frame", frame)
            #cv2.resizeWindow("Frame", 2560, 1440)
            
            #save_path = f"output/frame_{frame_jdx:05d}.png"
            #cv2.imwrite(save_path, frame)
            
            #key = cv2.waitKey(100) & 0xFF
                                            
            #if key == ord('q'):
            #    print("Exiting visualization.")
            #    break

        frame_idx += 1           
     
    cap.release()
    cv2.destroyAllWindows()  
    
    ade = ade / len(keep_frames)
    fde = fde / len(keep_frames)
    print(f"Evaluation --> ADE: {ade:.4f}, FDE: {fde:.4f}")

if __name__ == '__main__':  

    p = argparse.ArgumentParser(description='plot the annotations')
    p.add_argument('past_trajectory', type=int, help='Past Trajectory')
    p.add_argument('future_trajectory', type=int, help='Prediction Horizon')
    p.add_argument('video_path', type=str, help='Video Path')
    p.add_argument('annotations_file', type=str, help='Annotations directory')
    p.add_argument('--setting', default="gt", type=str, choices=['gt', 'trained'], 
                   help="Choose between visualizing ground truth intentions or predicted intentions.")
    p.add_argument('--checkpoint', type=str, default=None, help='Path to the trained model checkpoint.')
    p.add_argument('--device', type=str, default='cuda:1', choices=['cuda', 'cpu'], help='Device to run the model.')
    args = p.parse_args()

    trajectories = group_annotations_by_frame(args.past_trajectory, args.future_trajectory, args.annotations_file)
    if args.setting == "gt":
            plot_predictions_from_annotations(args.video_path, trajectories)
    else:
        if args.checkpoint is None:
            print("To run evaluation, please provide a checkpoint file.")
            exit()
        #predictor = RNNPredictor(args.past_trajectory, args.future_trajectory, args.device, checkpoint_file=args.checkpoint)
        #predictor = GCNLSTMPredictor(args.past_trajectory, args.future_trajectory, args.device, checkpoint_file=args.checkpoint)
        #predictor = GATLSTMPredictor(args.past_trajectory, args.future_trajectory, args.device, checkpoint_file=args.checkpoint)
        #predictor = GCNPredictor(args.past_trajectory, args.future_trajectory, args.device, checkpoint_file=args.checkpoint)
        predictor = GATPredictor(args.past_trajectory, args.future_trajectory, args.device, checkpoint_file=args.checkpoint)
        plot_predictions(args.video_path, trajectories, predictor)

