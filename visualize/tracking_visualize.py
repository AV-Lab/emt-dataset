import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
import argparse

GT_OBJECT_CLASSES = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Motorbike",
    3: "Small_motorised_vehicle",
    4: "Car",
    5: "Medium_vehicle",
    6: "Large_vehicle",
    7: "Bus",
    8: "Emergency_vehicle",
}


def load_gt_detections(gt_file, classes_of_interest=None):
    """
    Load ground truth detections in KITTI format, filtering by classes of interest.
    
    Args:
        gt_file (str): Path to the ground truth annotation file.
        classes_of_interest (set, optional): Set of class IDs to filter detections. If None, all classes are loaded.
    
    Returns:
        dict: Dictionary with frame_id as key and values as [x1, y1, x2, y2, score, class_id, track_id].
    """
    print(f"Loading GT file: {gt_file}")
    gt_detections = {}
    class_name_to_id = {name.lower(): id for id, name in GT_OBJECT_CLASSES.items()}

    try:
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:
                    continue

                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                class_name = parts[2].lower()
                class_id = class_name_to_id.get(class_name)
                if class_id is None or (classes_of_interest and class_id not in classes_of_interest):
                    continue

                x1, y1, x2, y2 = map(float, parts[6:10])
                score = float(parts[-1]) if len(parts) > 17 else 1.0
                
                gt_detections.setdefault(frame_id, []).append(
                    [x1, y1, x2, y2, score, class_id, track_id]
                )
    
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"Loaded {len(gt_detections)} frames of ground truth data")
    return gt_detections


def process_video(video_folder, gt_folder, classes_of_interest=None):
    """
    Process videos and overlay ground truth detections on frames.
    
    Args:
        video_folder (str): Path to the folder containing video frames.
        gt_folder (str): Path to the folder containing KITTI format annotations.
        classes_of_interest (set, optional): Set of class IDs to display. If None, all classes are displayed.
    """
    print("Tracking classes:", [GT_OBJECT_CLASSES[c] for c in classes_of_interest] if classes_of_interest else "All classes")
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    items = sorted(d for d in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, d)))
    
    for item in items:
        item_name = os.path.splitext(item)[0]
        print(f"\nProcessing {item_name}")

        frame_folder = os.path.join(video_folder, item)
        frame_files = sorted(os.listdir(frame_folder))
        if not frame_files:
            continue

        first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
        height, width = first_frame.shape[:2]
        total_frames = len(frame_files)
        
        gt_file = os.path.join(gt_folder, f"{item_name}.txt") if gt_folder else None
        if gt_file and not os.path.exists(gt_file):
            print(f"Warning: GT file not found for {item_name} searched at {gt_file}")
            continue
        
        gt_detections = load_gt_detections(gt_file, classes_of_interest)
        if gt_detections is None:
            continue
        
        with tqdm(total=total_frames, desc=f"Processing {item_name}") as pbar:
            for frame_counter, frame_file in enumerate(frame_files, start=1):
                frame_path = os.path.join(frame_folder, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                frame = cv2.resize(frame, (width, height))
                
                if frame_counter in gt_detections:
                    frame_dets = np.array(gt_detections[frame_counter])
                    dets = frame_dets[:, :4]  # x1, y1, x2, y2
                    class_ids = frame_dets[:, 5].astype(int)
                    track_ids = frame_dets[:, 6].astype(int)
                    
                    detections = sv.Detections(
                        xyxy=dets,
                        confidence=np.ones(len(dets)),
                        class_id=class_ids,
                        tracker_id=track_ids,
                    )
                    labels = [f"#{tid} {GT_OBJECT_CLASSES[cid]}" for tid, cid in zip(track_ids, class_ids)]
                    
                    annotated_frame = box_annotator.annotate(frame, detections=detections)
                    label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
                    
                    if annotated_frame.shape[1] > 1280:
                        scale = 1280 / annotated_frame.shape[1]
                        annotated_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
                    
                    cv2.imshow("frame", annotated_frame)
                    if cv2.waitKey(40) & 0xFF == ord("q"):
                        break
                
                pbar.update(1)
    
    print("Processing complete.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_folder", type=str, default="../data/frames/", help="Path to video frames folder")
    parser.add_argument("--gt_folder", type=str, default="../data/annotations/tracking_annotations/kitti/", help="Path to ground truth annotations folder")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="List of class IDs to track")
    args = parser.parse_args()
    
    process_video(args.frames_folder, args.gt_folder, set(args.classes) if args.classes else None)
