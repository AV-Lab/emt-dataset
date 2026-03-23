# viz/base_prediction_visualizer.py

import os
import cv2


class BasePredictionVisualizer:
    @staticmethod
    def _draw(frame, data):
        for bbox, future_traj in data:
            if len(future_traj) == 0:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for i in range(1, len(future_traj)):
                start_pt = (int(future_traj[i - 1][0]), int(future_traj[i - 1][1]))
                end_pt = (int(future_traj[i][0]), int(future_traj[i][1]))
                cv2.line(frame, start_pt, end_pt, (0, 0, 255), 2)
                cv2.circle(frame, start_pt, 3, (0, 0, 255), -1)

            if len(future_traj) >= 2:
                cv2.arrowedLine(
                    frame,
                    (int(future_traj[-2][0]), int(future_traj[-2][1])),
                    (int(future_traj[-1][0]), int(future_traj[-1][1])),
                    (0, 0, 255),
                    2,
                    tipLength=0.3,
                )
            else:
                cv2.circle(
                    frame,
                    (int(future_traj[-1][0]), int(future_traj[-1][1])),
                    5,
                    (0, 0, 255),
                    -1,
                )

        return frame

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
            img = BasePredictionVisualizer._draw(img, objs)

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