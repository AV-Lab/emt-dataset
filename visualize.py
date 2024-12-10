#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: murdism
"""

import cv2
import sys
import os
import argparse

def draw_bounding_boxes(image, annotations, frame_idx):
    """
    Draw bounding boxes on the given image based on the annotations for the current frame.
    """
    
    for ann in annotations:
        frame, track_id, obj_type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, *_ = ann
       

        # Process only the current frame
        if int(frame) != frame_idx:
            print (f"Invalid frame {int(frame) , frame_idx}")
            continue

        # Convert bounding box coordinates to integers
        bbox_left, bbox_top, bbox_right, bbox_bottom = map(lambda x: int(float(x)), [bbox_left, bbox_top, bbox_right, bbox_bottom])

        # Draw the 2D bounding box
        color = (0, 255, 0)  # Green color
        cv2.rectangle(image, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color, 2)

        # Add label (object type and track ID)
        label = f"{obj_type} " #({track_id})
        cv2.putText(image, label, (bbox_left, bbox_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image
# def process_video(video_path, annotation_path, save_output=False, output_video_path="output_video.mp4"):
#     """
#     Process a video to overlay KITTI annotations, visualize, and optionally save the result.
#     Annotations start from second frame and repeat every third frame.
#     """
#     # Read annotations from file
#     with open(annotation_path, 'r') as f:
#         annotations = [line.strip().split() for line in f.readlines()]

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Unable to open video file.")
#         return

#     # Define video writer if saving output
#     writer = None
#     if save_output:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
#     frame_idx = 0
#     last_annotation_frame = None
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video!")
#             break

#         # Only get new annotations on frames that should be annotated (1, 4, 7, etc.)
#         if frame_idx % 3 == 1:
#             annotation_idx = frame_idx // 3 + 1
#             current_frame_annotations = [ann for ann in annotations if int(ann[0]) == annotation_idx]
#             if current_frame_annotations:
#                 last_annotation_frame = current_frame_annotations
        
#         # Use the last valid annotations for visualization
#         if last_annotation_frame:
#             annotated_frame = draw_bounding_boxes(frame, last_annotation_frame, frame_idx // 3 + 1)
#         else:
#             annotated_frame = frame

#         # Save frame if requested
#         if save_output and writer:
#             writer.write(annotated_frame)

#         # Display frame
#         cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
#         cv2.imshow("Frame", annotated_frame)
#         cv2.resizeWindow("Frame", 1280, 720)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             print("Exiting visualization.")
#             break

#         frame_idx += 1

#     # Release resources
#     cap.release()
#     if writer:
#         writer.release()
#     cv2.destroyAllWindows()
def process_video(video_path, annotation_path, save_output=False, output_video_path="output_video.mp4"):
    """
    Process a video to overlay KITTI annotations, visualize, and optionally save the result.
    Handles both 30fps and 25fps videos for 10Hz annotations.
    """
    # Read annotations from file
    with open(annotation_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]
        print(f"Loaded {len(annotations)} annotations")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {frame_width}x{frame_height}")

    # Calculate frame interval based on FPS to get 10Hz
    if fps == 30:
        interval = 3  # Every 3rd frame
        frame_start = 1  # Second frame
    elif fps == 25:
        interval = 2.5  # Every 2.5 frames
        frame_start = 1  # Second frame
    else:
        print(f"WARNING: Unexpected FPS: {fps}")
        interval = fps / 10
        frame_start = 1

    # Define video writer if saving output
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # frame_idx = 0
    # last_annotation_frame = None
    # next_annotated_frame = frame_start
    # annotation_counter = 0
    # annotation_idx = 0
    
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print(f"End of video! Processed {frame_idx} frames with {annotation_counter} annotations")
    #         break
        
    #     # Check if current frame should have annotation
    #     if frame_idx == int(next_annotated_frame):
    #         annotation_idx +=1 #int((frame_idx - frame_start) / interval) + 1
    #         current_frame_annotations = [ann for ann in annotations if int(ann[0]) == annotation_idx]
    #         if current_frame_annotations:
    #             last_annotation_frame = current_frame_annotations
    #             annotation_counter += len(current_frame_annotations)
    #             current_ann_idx = annotation_idx
               
    #         else:
    #             current_ann_idx = annotation_idx-1
    #         next_annotated_frame += interval
    frame_idx = 0
    last_annotation_frame = None
    next_annotated_frame = frame_start
    annotation_counter = 0
    annotation_idx = 0
    current_ann_idx = 0
    
    # For debugging
    last_valid_annotation_idx = 0
    
    while True:
    
        ret, frame = cap.read()
        if not ret:
            print(f"End of video! Processed {frame_idx} frames with {annotation_counter} annotations")
            break
        
        # Check if current frame should have annotation
        if frame_idx == int(next_annotated_frame):
            annotation_idx += 1
            current_frame_annotations = [ann for ann in annotations if int(ann[0]) == annotation_idx]
            
            if current_frame_annotations:
                print(f"Current annotation: {annotation_idx}")
                last_annotation_frame = current_frame_annotations
                annotation_counter += len(current_frame_annotations)
                last_valid_annotation_idx = annotation_idx  # Keep track of last valid annotation
            
            next_annotated_frame += interval
        print(f"current frame: {frame_idx}")
        
        # Create copy of frame for annotation
        display_frame = frame.copy()
        
        # Use the last valid annotations for visualization
        if last_annotation_frame:
            display_frame = draw_bounding_boxes(display_frame, last_annotation_frame, last_valid_annotation_idx)

        # Add debug information to frame
        debug_info = [
            f"FPS: {fps}",
            f"Frame: {frame_idx}/{total_frames}",
            f"Interval: {interval}",
            f"Annotations: {annotation_counter}",
            f"Ann idx: {last_valid_annotation_idx}"
        ]
        for i, text in enumerate(debug_info):
            cv2.putText(display_frame, text, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame if requested
        if save_output and writer:
            writer.write(display_frame)

        # Display frame
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", display_frame)
        cv2.resizeWindow("Frame", 1280, 720)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            print("Exiting visualization.")
            break
        elif key == ord('s'):  # Add ability to save current frame
            cv2.imwrite(f"frame_{frame_idx:06d}.jpg", display_frame)
            print(f"Saved frame_{frame_idx:06d}.jpg")

        frame_idx += 1

    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
def main():
    # Set up argument parser
    p = argparse.ArgumentParser(description='Visualize and optionally save annotated video')
    p.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    p.add_argument('--annotation_path', type=str, help='Path to the annotation file. If not provided, will be inferred from video path')
    p.add_argument('-save', action='store_true', help='Flag to save the annotated video')
    
    args = p.parse_args()

    # Get video path from arguments
    video_path = args.video_path
    
    # If annotation path not provided, infer it from video path
    if args.annotation_path:
        annotation_path = args.annotation_path
    else:
        # Extract video number and construct annotation path
        video_number = os.path.splitext(os.path.basename(video_path))[0]  # Gets filename without extension
        annotation_path = f"emt/kitti_annotations/{video_number}.txt"
    
    # Handle output path if saving is enabled
    if args.save:
        # Create output directory if it doesn't exist
        output_dir = "emt/annotated_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct output path
        video_name = os.path.basename(video_path)
        output_video_path = os.path.join(output_dir, video_name)
        
        print(f"Saving annotated video to {output_video_path}")
        process_video(video_path, annotation_path, save_output=True, output_video_path=output_video_path)
    else:
        print("Visualizing annotated video without saving.")
        process_video(video_path, annotation_path, save_output=False)

if __name__ == "__main__":
    main()