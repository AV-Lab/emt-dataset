# from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os
import argparse
# import motmetrics as mm
import pandas as pd
import traceback
# from scipy.optimize import linear_sum_assignment
import time
# import torch
from tqdm import tqdm
from tracker_utils import print_args,DetectionSource,TrackerConfig,Trackers,load_gt_detections,CLASSES_OF_INTEREST,save_to_kitti_format,load_detector,get_detections,handle_visualization

import warnings
warnings.filterwarnings('ignore')


def process_batch(frames_batch, frame_indices, detection_type, detector, gt_detections, tracker,tracker_name, 
                 height, width, use_gt_tracks, results_file, frame_offset):
    """Process a batch of frames and return detections and tracking data"""
    all_detections = []
    batch_tracking_data = []  # Store tracking data for batch writing
    
    for frame, frame_idx in zip(frames_batch, frame_indices):
        dets, class_ids, gt_track_ids = get_detections(frame, frame_idx, detection_type, detector, gt_detections)
        # print(f"dets before: {dets.shape}")
        
        if dets is not None:
            if use_gt_tracks:
                # Use ground truth tracks directly
                detections = sv.Detections(
                    xyxy=dets[:,:4],
                    confidence=np.ones(len(dets)),
                    class_id=class_ids,
                    tracker_id=gt_track_ids
                )
                batch_tracking_data.append((frame_idx, detections))
                all_detections.append((frame, detections))
            else:
                # Use tracker
                if tracker_name=="byte":
                    online_targets = tracker.update(dets, [height, width], [height, width])
                elif tracker_name=="bot":
                    try:
                        online_targets = tracker.update(dets, frame)
                    except Exception as e:
                        print(f"Error {str(e)} in frame {frame_idx}")
                        # print(f" Frame [height, width] : {height, width,np.array(frame).shape}\ndets in frame:{np.max(dets)}: {str(e)}")
                        continue
                if len(online_targets) > 0:
                    track_boxes = np.array([t.tlwh for t in online_targets])
                    track_ids = np.array([t.track_id for t in online_targets])
                    track_scores = np.array([t.score for t in online_targets])
                    track_clases = np.array([int(t.class_id) for t in online_targets])
                    # print(f"track_clases:{track_clases}")
                    
                    track_boxes_xyxy = np.column_stack([
                        track_boxes[:, 0],
                        track_boxes[:, 1],
                        track_boxes[:, 0] + track_boxes[:, 2],
                        track_boxes[:, 1] + track_boxes[:, 3]
                    ])

                    detections = sv.Detections(
                        xyxy=track_boxes_xyxy,
                        confidence=track_scores,
                        class_id=track_clases,
                        tracker_id=track_ids
                    )
                    batch_tracking_data.append((frame_idx, detections))
                    all_detections.append((frame, detections))
                    # print(f"dets after: {len(detections)}")
                else:
                    all_detections.append((frame, None))
                    # print(f"dets after: {0}")
        else:
            all_detections.append((frame, None))
    
    # Save all detections from batch at once
    for frame_idx, detections in batch_tracking_data:
        save_to_kitti_format(detections, frame_idx, results_file, frame_offset)
    
    return all_detections
def load_frame_batch(frame_counter, batch_size, total_frames, frames, frame_folder, cap, frame_files=None):
    """Load a batch of frames"""
    frames_batch = []
    frame_indices = []
    for i in range(batch_size):
        if frame_counter + i > total_frames:
            break
            
        if frames:
            frame = cv2.imread(os.path.join(frame_folder, frame_files[frame_counter + i - 1]))
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        if frame is not None:
            frames_batch.append(frame)
            frame_indices.append(frame_counter + i)
            
    return frames_batch, frame_indices


def process_detection(detector,video_path, output_path, frames, frame_folder, frame_offset,
                     gt_file, detection_type,use_gt_tracks, show_display, 
                     save_video, batch_size, results_path,tracker_name="byte"):
    """Process video/frames and save tracking results"""
    video_name = os.path.basename(frame_folder.rstrip('/')) if frames else os.path.splitext(os.path.basename(video_path))[0]
    
    gt_detections = load_gt_detections(gt_file) if detection_type == DetectionSource.GT else None
   

    # Get video properties
    if frames:
        frame_files = sorted(os.listdir(frame_folder))
        first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
        height, width = first_frame.shape[:2]
        total_frames = len(frame_files)
        fps = 10
    else:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize tracker components
    tracker_folder = Trackers.get_tracker_folder(tracker_name, use_gt_tracks)
    try:
        if use_gt_tracks:
            tracker = None
            print("\nUsing ground truth tracks - No Tracker Initialized")
        else:
            # print(f"Initializing {tracker_folder}")
            try:
                tracker_inst = Trackers(tracker_name, fps)
                tracker = tracker_inst.get_tracker()
                # print(f"Successfully initialized {tracker_folder} with {fps} FPS")
            except NotImplementedError as e:
                print(f"Tracker not implemented: {tracker_folder} with {fps} FPS")
                raise e
            except Exception as e:
                print(f"Failed to initialize {tracker_folder}: {str(e)}")
                raise e

    except Exception as e:
        print(f"Error in tracker initialization: {e}")
        return

    
    box_annotator = sv.BoxAnnotator() if (show_display or save_video) else None
     # label_annotator = sv.BoxAnnotator(text_thickness=1, text_scale=0.5) if (show_display or save_video) else None
    if tracker_name=="byte":
        label_annotator = sv.LabelAnnotator() if (show_display or save_video) else None
        print(f"Added Label Annotator for classes and tarck id!")
    else:
        label_annotator = None
        print(f"No Label Annotator!")

    # Initialize output
    results_file = open(results_path, 'w')
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if save_video else None
    
    # Process frames
    frame_counter = 1
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while frame_counter <= total_frames:
            
            frames_batch, frame_indices = load_frame_batch(
                frame_counter, batch_size, total_frames, frames, frame_folder, 
                None if frames else cap, frame_files if frames else None
            )
            
            if not frames_batch:
                break

            processed_frames = process_batch(
                frames_batch, frame_indices, detection_type, detector, gt_detections,
                tracker,tracker_name, height, width, use_gt_tracks, results_file, frame_offset
            )

            if show_display or save_video:
                for frame, detections in processed_frames:
                    if handle_visualization(frame, detections, detection_type, 
                                         box_annotator, writer, show_display,label_annotator):
                        break

            pbar.update(len(frames_batch))
            frame_counter += len(frames_batch)

    # Cleanup
    results_file.close()
    if not frames:
        cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()

def process_datatset(video_folder, output_folder, gt_folder=None, use_frames=False, 
                        detection_type=DetectionSource.Yolo_off_shelf,tracker_name='ByteTracker', use_gt_tracks=False, 
                        show_display=False, save_video=False, batch_size=32):
    

    
    # Get detector folder name
    detector_folder = DetectionSource.get_detector_name(detection_type)
    print(f"detector_folder: {detector_folder}")
    if detection_type == "Yolo_fine_tunned":
        detection_type = DetectionSource.Yolo_fine_tunned
    elif detection_type == "Yolo_off_shelf":
        detection_type = DetectionSource.Yolo_off_shelf
    else:
        detection_type == DetectionSource.GT


    # Initialize detector
    detector = load_detector(detection_type) if detection_type == DetectionSource.Yolo_fine_tunned or  detection_type == DetectionSource.Yolo_off_shelf else None


    """Process videos and save results in TrackEval format"""  
    tracker_folder = Trackers.get_tracker_folder(tracker_name, use_gt_tracks)

    # Create tracker-specific output folders
    pred_data_folder = os.path.join(output_folder, tracker_folder,detector_folder, "tracked_predictions")
    os.makedirs(pred_data_folder, exist_ok=True)
    
    track_video_folder = os.path.join(output_folder, tracker_folder,detector_folder, "tracked_videos")
    os.makedirs(track_video_folder, exist_ok=True)

    # Get list of input files
    all_items = os.listdir(video_folder)
    if use_frames:
        items = sorted([d for d in all_items if os.path.isdir(os.path.join(video_folder, d))])
    else:
        items = sorted([f for f in all_items if f.endswith(('.mp4', '.avi', '.mov'))])

    print(f"\nProcessing {len(items)} {'frame folders' if use_frames else 'videos'}")
    

    for item in items:
        item_name = item.split('.')[0]
        print(f"\nProcessing {item_name}")
        
        # Setup paths
        if use_frames:
            frame_folder = os.path.join(video_folder, item)
            video_path = None
            output_video = os.path.join(track_video_folder, f"{item_name}_tracked.mp4") if save_video else None
        else:
            video_path = os.path.join(video_folder, item)
            frame_folder = None
            output_video = os.path.join(track_video_folder, f"{item_name}_tracked.mp4") if save_video else None

        # Setup output file for tracking results
        results_path = os.path.join(pred_data_folder, f"{item_name}.txt")
        
        # Get GT file path from gt_folder
        gt_file = os.path.join(gt_folder, f"{item_name}.txt") if gt_folder else None
        
        if gt_file and not os.path.exists(gt_file):
            print(f"Warning: GT file not found for {item_name}")
        

        try:
            process_detection(
                detector,
                video_path=video_path,
                output_path=output_video,
                frames=use_frames,
                frame_folder=frame_folder,
                frame_offset=0,
                gt_file=gt_file,
                detection_type=detection_type,
                use_gt_tracks=use_gt_tracks,
                show_display=show_display,
                save_video=save_video,
                batch_size=batch_size,
                results_path=results_path,
                tracker_name=tracker_name
                
            )
        except Exception as e:
            print(f"Error processing {item_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\nResults saved in TrackEval format to: {output_folder}")
    return output_folder



def run_process(args, detection_type):
    """Run the video processing with specified detection type"""
    # Update detection type
    args.det = detection_type
    
    # Print updated arguments
    print_args(args, args.verbose)
    print(f"\nStarting batch processing with {detection_type}...")
    
    t0 = time.time()
    process_datatset(
        video_folder=args.input,
        output_folder=args.output,
        gt_folder=args.gt_folder,
        use_frames=args.use_frames,
        detection_type=detection_type,
        tracker_name=args.tracker_name,
        use_gt_tracks=args.use_gt_tracks,
        show_display=args.show_display,
        save_video=args.save_video,
        batch_size=args.batch_size
    )
    
    t1 = time.time()
    total_time = t1 - t0
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    time_spent = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
    print(f"\nBatch Processing Statistics for {detection_type}:")
    print("-" * 50)
    print(f"Total processing time: {time_spent}")
    print("-" * 50)

if __name__ == "__main__":

    # Create parser with print_args functionality built in
    parser = argparse.ArgumentParser(description='Process videos with tracking')
    parser.add_argument('--input', type=str, default='emt/frames/', 
                    help='Path to input folder (videos or frame folders)')
    parser.add_argument('--output', type=str,  default="Trackers/", 
                    help='Path to output folder')
    parser.add_argument('--tracker_name', type=str,  default="byte", 
                    help='Tracker name use byte, bot or boost')
    parser.add_argument('--gt_folder', type=str, default='emt/emt_annotations/labels_full/',
                    help='Path to folder containing GT files')
    parser.add_argument('--use_frames', default=True,action='store_true', 
                    help='Input folder contains frame folders instead of videos')
    parser.add_argument('--det', type=str, choices=['Yolo_fine_tunned','Yolo_off_shelf', 'gt'], default='gt', 
                    help='Detection source')
    parser.add_argument('--use_gt_tracks', action='store_true', 
                    help='Use ground truth detections and tracking')
    parser.add_argument('--show_display', action='store_true', 
                    help='Show visualization')
    parser.add_argument('--save_video', action='store_true', 
                    help='Save output videos')
    parser.add_argument('--batch_size', type=int, default=1, 
                    help='Batch size for processing')
    parser.add_argument('--verbose', default=True,action='store_true',
                    help='Enable verbose output mode')
    
    
    args = parser.parse_args()
    
    # print_args(args, args.verbose)  # Print configuration settings
    args = parser.parse_args()
    
    # # First run with original detection type
    run_process(args, args.det)
    
    # Second run with Yolo_off_shelf
    run_process(args, 'Yolo_off_shelf')

    # Third run with Yolo_fine_tunned
    run_process(args, 'Yolo_fine_tunned')

 