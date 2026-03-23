import os
import cv2
import numpy as np
import supervision as sv


class BaseBBoxVisualizer:
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    @staticmethod
    def _render(video_path: str, dets, scale: float = 1.0):
        frames = [os.path.join(video_path, img) for img in sorted(os.listdir(video_path))]

        title = "video"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

        num_frames = len(frames)

        for idx, frame in enumerate(frames):
            img = cv2.imread(frame)
            if img is None:
                continue

            objs = dets[idx] if idx < len(dets) else []

            if len(objs) > 0:
                boxes = np.array([o[:4] for o in objs], dtype=float)
                cids = np.array([o[4] for o in objs], dtype=int)
                detections = sv.Detections(xyxy=boxes, class_id=cids)

                labels = [o[5] for o in objs]

                img = BaseBBoxVisualizer.box_annotator.annotate(img, detections)
                img = BaseBBoxVisualizer.label_annotator.annotate(img, detections, labels)

            frame_name = os.path.basename(frame)
            text = f"Frame {idx+1}/{num_frames}  ({frame_name})"
            cv2.putText(
                img,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            cv2.imshow(title, img)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()