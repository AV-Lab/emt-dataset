
import os
import json
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from .utils import parse_split_metadata
from config.labels import AGENT_LABELS
from typing import List


import os
import json
from typing import List, Dict, Any

from config.labels import AGENT_LABELS


def generate_detection_annotations(videos: dict[str, tuple[int, int]],
                                   annotations_dir: str,
                                   output_json: str):

    categories = dict(list(AGENT_LABELS.items())[:9])

    coco_data = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [
            {
                "id": class_id,
                "name": class_name,
                "supercategory": "person" if class_name in ["Pedestrian", "Cyclist"] else "vehicle",
            }
            for class_name, class_id in categories.items()
        ],
    }

    track_ids = {}
    next_track_id = 1
    image_id = 1
    ann_id = 1
    video_id = 1

    for vn, (w, h) in videos.items():
        v_ann_dir = os.path.join(annotations_dir, vn)

        coco_data["videos"].append({"id": video_id, "folder_name": vn})
        frame_files = sorted(
            [f for f in os.listdir(v_ann_dir) if f.lower().endswith(".json")]
        )

        for frame_file in frame_files:
            ann_path = os.path.join(v_ann_dir, frame_file)
            frame_stem = frame_file.split('.')[0][2:]
            frame_id = int(frame_stem)

            coco_data["images"].append({
                "file_name": os.path.join(vn, f"{frame_stem}.jpg"),
                "id": image_id, #global unique id
                "frame_id": frame_id,
                "video_id": video_id,
                "height": h,
                "width": w,
                "has_annotation": True,
            })

            with open(ann_path, "r") as f:
                frame_data = json.load(f)
                
            instances = frame_data[0]["instances"]
            for inst in instances:
                agent_type = inst["classValues"][0]["value"]
                agent_type = agent_type.replace(" ", "_")

                if agent_type not in categories: continue

                # Extract the box coordinates    
                points = inst.get("contour", {}).get("points", [])
                xs = [float(p["x"]) for p in points]
                ys = [float(p["y"]) for p in points]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                # Extract the tracklet ID
                track_id_str = inst["trackId"]
                if (track_id_str, agent_type) not in track_ids:
                    track_ids[(track_id_str, agent_type)] = next_track_id
                    next_track_id += 1
                track_id = track_ids[(track_id_str, agent_type)]

                coco_data["annotations"].append({
                    "id": ann_id,
                    "category_id": categories[agent_type],
                    "image_id": image_id,
                    "track_id": track_id,
                    "track_id_str": track_id_str,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "occluded": 0,
                    "truncated": 0.0,
                    "alpha": 0.0,
                })
                ann_id += 1

            image_id += 1

        print(f"Video #id {video_id} #name {vn} processed")
        video_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_data, f)

    track_mapping_path = output_json.replace(".json", "_mapping.json")
    with open(track_mapping_path, "w") as f:
        json.dump({f"{k[0]}|||{k[1]}": v for k, v in track_ids.items()}, f)

    print(
        f"Saved {output_json}: "
        f"{len(coco_data['images'])} images, "
        f"{len(coco_data['annotations'])} annotations"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating detection annotations in COCO format')
    parser.add_argument('--dataset_dir', type=str, help='dataset folder path')
    args = parser.parse_args()
    
    annotations_dir = os.path.join(args.dataset_dir, "raw_annotations")
    metadata_path = os.path.join(args.dataset_dir, "metadata.txt")
    train_videos, test_videos = parse_split_metadata(metadata_path)
    
    print(f"Total: {len(train_videos)} train videos")
    print(f"Total: {len(test_videos)} test videos")
    
    output_dir = os.path.join(args.dataset_dir, "annotations/detections")
    os.makedirs(output_dir, exist_ok=True)
    fouts = [os.path.join(output_dir, "train.json"), 
             os.path.join(output_dir, "test.json")]
    
    for vids, fout in zip([train_videos, test_videos], fouts):
        generate_detection_annotations(vids, annotations_dir, fout)
        