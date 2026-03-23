import os
import json
import cv2
import numpy as np
import supervision as sv

JSON_PATH = "/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/emt/emt-dataset/data/annotations/detections/test.json"
FRAMES_ROOT = "/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/emt/emt-dataset/data/frames"   # folder that contains video_5/
SCALE = 0.5

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    anns_by_image = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        anns_by_image.setdefault(image_id, []).append(ann)

    ordered_images = sorted(data["images"], key=lambda x: x["id"])
    return ordered_images, anns_by_image, categories


def main():
    images, anns_by_image, categories = load_coco_annotations(JSON_PATH)

    title = "detections"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    num_frames = len(images)

    for idx, img_info in enumerate(images):
        img_path = os.path.join(FRAMES_ROOT, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        anns = anns_by_image.get(img_info["id"], [])
        objs = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            cid = ann["category_id"]
            label = categories[cid]
            objs.append([x1, y1, x2, y2, cid, label])

        if len(objs) > 0:
            boxes = np.array([o[:4] for o in objs], dtype=float)
            cids = np.array([o[4] for o in objs], dtype=int)
            labels = [o[5] for o in objs]

            detections = sv.Detections(xyxy=boxes, class_id=cids)
            img = box_annotator.annotate(img, detections)
            img = label_annotator.annotate(img, detections, labels)

        frame_name = os.path.basename(img_info["file_name"])
        text = f"Frame {idx+1}/{num_frames} ({frame_name})"
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

        if SCALE != 1.0:
            img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(title, img)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()