import json

from viz.base_intention_visualizer import BaseIntentionVisualizer


class IntentionVisualizer(BaseIntentionVisualizer):
    @staticmethod
    def _load_annotations(annotations_file):
        dets_dict = {}

        with open(annotations_file, "r") as f:
            data = json.load(f)

        for _, track in data.items():
            frames = track["frames"]
            bboxes = track["bbox"]
            intentions = track["intention"]

            for frame_id, bbox, intention in zip(frames, bboxes, intentions):
                dets_dict.setdefault(frame_id, []).append((bbox, intention))

        if not dets_dict:
            return []

        max_frame = max(dets_dict.keys())
        dets = []
        for frame_id in range(1, max_frame + 1):
            dets.append(dets_dict.get(frame_id, []))

        return dets

    @staticmethod
    def visualize(video_path, annotations_file, scale: float = 1.0):
        dets = IntentionVisualizer._load_annotations(f"{annotations_file}.json")
        IntentionVisualizer._render(video_path, dets, scale)