# viz/prediction_visualizer.py

import json

from viz.base_prediction_visualizer import BasePredictionVisualizer


def bbox_to_xy(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


class PredictionVisualizer(BasePredictionVisualizer):
    @staticmethod
    def _load_annotations(annotations_file, traj_len):
        dets_dict = {}

        with open(annotations_file, "r") as f:
            data = json.load(f)

        for _, track in data.items():
            frames = track["frames"]
            bboxes = track["bbox"]
            positions = [bbox_to_xy(bbox) for bbox in bboxes]

            for i, frame_id in enumerate(frames):
                bbox = bboxes[i]

                r = min(i + traj_len + 1, len(frames))
                future_traj = positions[i + 1:r]

                dets_dict.setdefault(frame_id, []).append((bbox, future_traj))

        if not dets_dict:
            return []

        max_frame = max(dets_dict.keys())
        dets = []
        for frame_id in range(1, max_frame + 1):
            dets.append(dets_dict.get(frame_id, []))

        return dets

    @staticmethod
    def visualize(video_path, annotations_file, traj_len, scale: float = 1.0):
        dets = PredictionVisualizer._load_annotations(
            f"{annotations_file}.json",
            traj_len,
        )
        PredictionVisualizer._render(video_path, dets, scale)