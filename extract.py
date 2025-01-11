import numpy as np
import cv2
import argparse


class FramesExtractor:
    def __init__(self, base_dir, splits=['training']):
        self.base_dir = base_dir
        self.splits = splits
        
        # EMT directory structure
        self.videos_dir = os.path.join(base_dir, 'raw')
        self.annot_dir = os.path.join(base_dir, 'kitti_annotations')
        self.output_dir = os.path.join(base_dir, 'coco')
        self.frames_dir = os.path.join(self.output_dir, 'frames')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.categories = {
            'Pedestrian': 1,
            'Cyclist': 2,
            'Motorbike': 3,
            'Small_motorised_vehicle': 4,
            'Medium_vehicle': 5,
            'Large_vehicle': 6,
            'Car': 7,
            'Bus': 8,
            'Emergency_vehicle': 9
        }
        
        # Track ID mapping
        self.track_ids = {}
        self.next_track_id = 1

def extract_frames(vidname, videos_dir, outdir):
    """Extract frames using cv2, starting from second frame"""
    base_name = vidname.split('.')[0]
    video_file = os.path.join(videos_dir, vidname)
    images_dir = os.path.join(outdir, base_name)
    print(f"--------------------------------{images_dir}--------------------------------")
    
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return 0

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frame_idx = 0  # Current frame index
    save_idx = 1   # Index for saved frames (1-based)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Start from second frame and take every third frame
        if frame_idx % 3 == 1:  # This will get frames 1, 4, 7, etc.
            frame_path = os.path.join(images_dir, f"{save_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            save_idx += 1
            
        frame_idx += 1
    
    cap.release()
    
    # Verify extracted frames
    imglist = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    expected_frames = (total_frames - 2) // 3 + 1  # Starting from second frame, every third
    print(f"Expected frames: {expected_frames}, Extracted frames: {len(imglist)}")
    
    return len(imglist)