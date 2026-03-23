import os
import cv2


class BaseIntentionVisualizer:
    @staticmethod
    def _draw(frame, data):
        for bbox, intention in data:
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            text_size = cv2.getTextSize(intention, font, font_scale, thickness)[0]
            label_x1 = x1
            label_y1 = max(y1 - text_size[1] - 8, 0)
            label_x2 = x1 + text_size[0] + 8
            label_y2 = y1

            cv2.rectangle(
                frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                (0, 165, 255),
                -1,
            )
            cv2.putText(
                frame,
                intention,
                (label_x1 + 4, label_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
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
            img = BaseIntentionVisualizer._draw(img, objs)

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