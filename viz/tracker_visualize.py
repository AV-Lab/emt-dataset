import os
from typing import Optional, List

from config.labels import AGENT_LABELS
from viz.base_bbox_visualizer import BaseBBoxVisualizer


class TrackerVisualizer(BaseBBoxVisualizer):
    @staticmethod
    def _load_annotations(ann_file: str, classes: Optional[List[str]]):
        dets_dict = {}

        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue

                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                agent = parts[2].replace(" ", "_")

                if agent == "AV" or (classes and agent not in classes):
                    continue

                cid = AGENT_LABELS[agent]
                x1, y1, x2, y2 = map(float, parts[6:10])

                label = f"{agent}#{track_id}"
                dets_dict.setdefault(frame_id, []).append([x1, y1, x2, y2, cid, label])

        if not dets_dict:
            return []

        max_frame = max(dets_dict.keys())
        dets = []
        for frame_id in range(1, max_frame + 1):
            dets.append(dets_dict.get(frame_id, []))

        return dets

    @staticmethod
    def visualize(video_path: str, ann_file: str, classes=None, scale: float = 1.0):
        dets = TrackerVisualizer._load_annotations(f"{ann_file}.txt", classes)
        TrackerVisualizer._render(video_path, dets, scale)