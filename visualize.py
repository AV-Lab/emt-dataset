import argparse
import os
from viz.box_visualizer import BBoxVisualizer
from viz.tracker_visualize import TrackerVisualizer  
from viz.intention_visualize import IntentionVisualizer
from viz.predictor_visualize import PredictionVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--mode", 
                        type=str, 
                        choices=["dets", "preds", "intentions", "tracks"], 
                        default="dets",
                        help="What to visualize")
    parser.add_argument("--video_name", type=str, required=True, help="Video to visualize")
    parser.add_argument("--traj_len", type=int, default=10, help="Trajectory length to visualize")
    args = parser.parse_args()

    frames_dir = os.path.join(args.dataset_path, "frames")
    video_path = os.path.join(frames_dir, args.video_name)
    
    if args.mode == "dets":
        ann_path = os.path.join(args.dataset_path, f"raw_annotations/{args.video_name}")
        BBoxVisualizer.visualize(video_path, ann_path)
    
    elif args.mode == "tracks":
        ann_path = os.path.join(args.dataset_path, f"annotations/tracking/gmot/{args.video_name}")
        TrackerVisualizer.visualize(video_path, ann_path)        
    
    elif args.mode == "intentions":
        ann_path = os.path.join(args.dataset_path, f"annotations/intention/{args.video_name}")
        IntentionVisualizer.visualize(video_path, ann_path)   
        
    elif args.mode == "preds":
        ann_path = os.path.join(args.dataset_path, f"annotations/prediction/{args.video_name}")
        PredictionVisualizer.visualize(video_path, ann_path, traj_len=args.traj_len)   
        
    else:
        print("The specified mode is not supported")