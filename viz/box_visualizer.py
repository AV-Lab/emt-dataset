import os
import json
from typing import Optional, List

from config.labels import AGENT_LABELS
from viz.base_bbox_visualizer import BaseBBoxVisualizer


class BBoxVisualizer(BaseBBoxVisualizer):
    @staticmethod
    def _load_annotations(anns: List[str], classes: Optional[List[str]]):
        dets = []
        for ann in anns:
            with open(ann, "r") as f:
                data = json.load(f)
                frame_dets = []

                insts = data[0]["instances"]
                for inst in insts:
                    pts = inst["contour"]["points"]
                    xs = [p["x"] for p in pts]
                    ys = [p["y"] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    agent, loc, action = None, None, None
                    vals = inst["classValues"]
                    for val in vals:
                        if val["alias"] == "Agent":
                            agent = val["value"].replace(" ", "_")
                        elif val["alias"] == "Location":
                            loc = val["value"]
                        elif val["alias"] == "Action":
                            action = val["value"]

                    if agent == "AV" or (classes and agent not in classes):
                        continue

                    cid = AGENT_LABELS[agent]
                    label = f"{agent}#{loc}@{action}"
                    frame_dets.append([x1, y1, x2, y2, cid, label])

                dets.append(frame_dets)
        return dets

    @staticmethod
    def visualize(video_path: str, ann_path: str, classes=None, scale: float = 1.0):
        anns = [os.path.join(ann_path, ann) for ann in sorted(os.listdir(ann_path))]
        dets = BBoxVisualizer._load_annotations(anns, classes)
        BBoxVisualizer._render(video_path, dets, scale)