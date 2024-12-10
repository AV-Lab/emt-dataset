
import os
import json
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
import cv2
import argparse
from tqdm import tqdm

# def extract_frames(vidname, videos_dir, outdir):
#     """Extract frames using ffmpeg, starting from second frame"""
#     base_name = vidname.split('.')[0]  # Remove any extension (.mp4 or .MP4)
#     video_file = os.path.join(videos_dir, vidname)
#     images_dir = os.path.join(outdir, base_name)
#     print(f"--------------------------------{images_dir}--------------------------------")
    
#     if not os.path.isdir(images_dir):
#         os.makedirs(images_dir)

#     imglist = os.listdir(images_dir)
#     imglist = [img for img in imglist if img.endswith('.jpg')]

#     # if len(imglist) < 2:  # very few or no frames try extracting again
#     # Extract every third frame starting from frame 1 (second frame)
#     command = 'ffmpeg -i {} -vf "select=not(mod(n-1\,3))" -vsync vfr -q:v 1 {}/%05d.jpg'.format(
#         video_file, images_dir)
#     print('run', command)
#     os.system(command)
    
#     imglist = os.listdir(images_dir)
#     imglist = [img for img in imglist if img.endswith('.jpg')]
    
#     return len(imglist)

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
class EMTVideoToCOCO:
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

    def get_numeric_track_id(self, string_id):
        """Convert string track IDs to numeric IDs while maintaining consistency"""
        if string_id not in self.track_ids:
            self.track_ids[string_id] = self.next_track_id
            self.next_track_id += 1
        return self.track_ids[string_id]

    def convert(self):
        for split in self.splits:
            coco_data = {
                'images': [],
                'annotations': [],
                'videos': [],
                'categories': [
                    {
                        'id': v, 
                        'name': k,
                        'supercategory': 'person' if k in ['Pedestrian', 'Cyclist'] else 'vehicle'
                    } 
                    for k, v in self.categories.items()
                ]
            }
            
            image_cnt = 0
            ann_cnt = 0
            video_cnt = 0
            
            videofiles = os.listdir(self.videos_dir)
            videofiles = [vf for vf in videofiles if vf.lower().endswith('.mp4')]
            
            for video_name in tqdm(sorted(videofiles), desc="Processing videos"):
                video_cnt += 1
                video_base = video_name.split('.')[0]  # Remove any extension (.mp4 or .MP4)
                
                # Extract frames using ffmpeg
                num_frames = extract_frames(video_name, self.videos_dir, self.frames_dir)
                
                coco_data['videos'].append({
                    'id': video_cnt,
                    'file_name': video_name
                })
                
                # Process frames
                frames_dir = os.path.join(self.frames_dir, video_base)
                frame_list = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
                
                for frame_idx, frame_name in enumerate(frame_list):
                    frame_path = os.path.join(frames_dir, frame_name)
                    img = cv2.imread(frame_path)
                    height, width = img.shape[:2]
                    
                    # Original video frame index (1, 4, 7, etc.)
                    orig_frame_idx = (frame_idx * 3) + 1
                    
                    image_info = {
                        'file_name': os.path.join(video_base, frame_name),
                        'id': image_cnt + frame_idx + 1,
                        'frame_id': orig_frame_idx,
                        'video_id': video_cnt,
                        'height': height,
                        'width': width,
                        'has_annotation': True
                    }
                    coco_data['images'].append(image_info)
                
                # Process annotations
                ann_file = os.path.join(self.annot_dir, f'{video_base}.txt')
                if os.path.exists(ann_file):
                    with open(ann_file, 'r') as f:
                        annotations = [line.strip().split() for line in f.readlines()]
                    
                    # Process annotations frame by frame
                    for frame_idx in range(len(frame_list)):
                        annotation_idx = frame_idx + 1
                        current_frame_annotations = [
                            ann for ann in annotations if int(ann[0]) == annotation_idx
                        ]
                        
                        for ann in current_frame_annotations:
                            obj_type = ann[2]
                            if obj_type not in self.categories:
                                continue
                            
                            # Get numeric track ID while preserving string ID
                            string_track_id = ann[1]
                            numeric_track_id = self.get_numeric_track_id(string_track_id)
                            
                            # Convert coordinates to COCO format
                            bbox_left = float(ann[6])
                            bbox_top = float(ann[7])
                            bbox_right = float(ann[8])
                            bbox_bottom = float(ann[9])
                            width = bbox_right - bbox_left
                            height = bbox_bottom - bbox_top
                            
                            ann_cnt += 1
                            annotation = {
                                'id': ann_cnt,
                                'category_id': self.categories[obj_type],
                                'image_id': image_cnt + frame_idx + 1,
                                'track_id': numeric_track_id,
                                'track_id_str': string_track_id,  # Preserve original string ID
                                'bbox': [bbox_left, bbox_top, width, height],
                                'area': width * height,
                                'iscrowd': 0,
                                'occluded': int(ann[4]),
                                'truncated': float(ann[3]),
                                'alpha': float(ann[5])
                            }
                            coco_data['annotations'].append(annotation)
                
                image_cnt += len(frame_list)
                print(f'Processed video {video_name}: {len(frame_list)} frames')
            
            # Save annotations
            out_file = os.path.join(self.output_dir, f'{split}.json')
            with open(out_file, 'w') as f:
                json.dump(coco_data, f)
            
            print(f'Converted {split} split: {len(coco_data["images"])} images, '
                  f'{len(coco_data["annotations"])} annotations')
            
            # Save track ID mapping for reference
            track_mapping_file = os.path.join(self.output_dir, f'{split}_track_mapping.json')
            with open(track_mapping_file, 'w') as f:
                json.dump(self.track_ids, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Convert EMT videos and KITTI annotations to COCO format')
    parser.add_argument('base_dir', type=str,
                       help='EMT base directory containing raw/ and kitti_annotations/ subdirectories')
    parser.add_argument('--splits', nargs='+', default=['training'],
                       help='Splits to process')
    args = parser.parse_args()
    
    converter = EMTVideoToCOCO(args.base_dir, args.splits)
    converter.convert()

if __name__ == '__main__':
    main()