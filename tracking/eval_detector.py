import cv2
import torch
import numpy as np
import os
from tqdm import tqdm  
import sys
from yolox.exp import get_exp, Exp as MyExp
from collections import defaultdict


class DetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.metrics = {
            'Pedestrian': {'TP': 0, 'FP': 0, 'FN': 0},
            'Cyclist': {'TP': 0, 'FP': 0, 'FN': 0},
            'Motorbike': {'TP': 0, 'FP': 0, 'FN': 0},
            'Vehicle': {'TP': 0, 'FP': 0, 'FN': 0}
        }

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def evaluate_frame(self, frame_detections, ground_truth):
        """
        Evaluate detections for a single frame.
        
        Args:
            frame_detections: List of dictionaries with 'bbox', 'class', and 'label' from process_frames
            ground_truth: List of [x1, y1, x2, y2, score, class_id, track_id] from gt_detections
        """
        frame_metrics = {cls_name: {'TP': 0, 'FP': 0, 'FN': 0} 
                        for cls_name in self.metrics.keys()}
        
        # Convert ground truth format
        gt_by_class = defaultdict(list)
        for gt in ground_truth:
            gt_class = self.map_gt_class(gt[5])  # gt[5] is class_id
            gt_by_class[gt_class].append(gt[:4])  # bbox coordinates
            
        # Convert detections format
        det_by_class = defaultdict(list)
        for det in frame_detections:
            if isinstance(det, dict) and 'bbox' in det:
                # Map detection class to evaluation class
                eval_class = self.map_detection_class(det.get('class', ''))
                if eval_class:
                    det_by_class[eval_class].append(det['bbox'])
        
        # Evaluate each class
        for class_name in self.metrics.keys():
            gt_boxes = gt_by_class[class_name]
            det_boxes = det_by_class[class_name]
            
            if not gt_boxes and not det_boxes:
                continue
                
            if not det_boxes:
                frame_metrics[class_name]['FN'] += len(gt_boxes)
                continue
                
            if not gt_boxes:
                frame_metrics[class_name]['FP'] += len(det_boxes)
                continue

            # Create IoU matrix
            iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
            for i, det_box in enumerate(det_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    iou_matrix[i, j] = self.calculate_iou(det_box, gt_box)

            # Match detections to ground truth
            matched_gt = set()
            matched_det = set()
            
            while True:
                if len(matched_gt) == len(gt_boxes) or len(matched_det) == len(det_boxes):
                    break
                    
                unmatched_iou = iou_matrix.copy()
                unmatched_iou[list(matched_det), :] = -1
                unmatched_iou[:, list(matched_gt)] = -1
                i, j = np.unravel_index(unmatched_iou.argmax(), unmatched_iou.shape)
                
                if unmatched_iou[i, j] < self.iou_threshold:
                    break
                    
                matched_gt.add(j)
                matched_det.add(i)
                frame_metrics[class_name]['TP'] += 1

            # Count unmatched as FP/FN
            frame_metrics[class_name]['FP'] += len(det_boxes) - len(matched_det)
            frame_metrics[class_name]['FN'] += len(gt_boxes) - len(matched_gt)

        return frame_metrics

    def map_gt_class(self, class_id):
        """Map ground truth class ID to evaluation class."""
        gt_mapping = {
            0: 'Pedestrian',
            1: 'Cyclist',
            2: 'Motorbike',
            3: 'Vehicle',  # Small_motorised_vehicle
            4: 'Vehicle',  # Car
            5: 'Vehicle',  # Medium_vehicle
            6: 'Vehicle',  # Large_vehicle
            7: 'Vehicle',  # Bus
            8: 'Vehicle'   # Emergency_vehicle
        }
        return gt_mapping.get(class_id, 'Unknown')

    def map_detection_class(self, det_class):
        """Map detection class to evaluation class."""
        det_mapping = {
            'Pedestrian': 'Pedestrian',
            'Cyclist': 'Cyclist',
            # 'Bicycle': 'Cyclist',
            'Motorbike': 'Motorbike',
            # 'Motorbike Rider': 'Motorbike',
            'Vehicle': 'Vehicle',
            'Car': 'Vehicle',
            'Bus': 'Vehicle',
            'Truck': 'Vehicle'
        }
        return det_mapping.get(det_class, None)

    def update_metrics(self, frame_metrics):
        """Update overall metrics with frame metrics."""
        for cls_name in self.metrics:
            for metric in ['TP', 'FP', 'FN']:
                self.metrics[cls_name][metric] += frame_metrics[cls_name][metric]

    def calculate_final_metrics(self):
        """Calculate precision, recall, and F1-score for each class."""
        results = {}
        overall_metrics = {'TP': 0, 'FP': 0, 'FN': 0}

        for cls_name, metrics in self.metrics.items():
            TP = metrics['TP']
            FP = metrics['FP']
            FN = metrics['FN']
            
            overall_metrics['TP'] += TP
            overall_metrics['FP'] += FP
            overall_metrics['FN'] += FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results[cls_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'TP': TP,
                'FP': FP,
                'FN': FN
            }

        # Calculate overall metrics
        overall_TP = overall_metrics['TP']
        overall_FP = overall_metrics['FP']
        overall_FN = overall_metrics['FN']
        
        overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0
        overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        results['overall'] = {
            'Precision': overall_precision,
            'Recall': overall_recall,
            'F1': overall_f1,
            'TP': overall_TP,
            'FP': overall_FP,
            'FN': overall_FN
        }

        return results

def format_results(results):
    """Format results for pretty printing."""
    output = []
    output.append("\nDetection Evaluation Results")
    output.append("=" * 50)
    
    for cls_name, metrics in results.items():
        output.append(f"\n{cls_name.upper()}")
        output.append("-" * 20)
        output.append(f"Precision: {metrics['Precision']:.4f}")
        output.append(f"Recall: {metrics['Recall']:.4f}")
        output.append(f"F1-Score: {metrics['F1']:.4f}")
        output.append(f"True Positives: {metrics['TP']}")
        output.append(f"False Positives: {metrics['FP']}")
        output.append(f"False Negatives: {metrics['FN']}")
    
    return "\n".join(output)

class UnifiedVideoDetector:
    def __init__(self, model_path, exp_file=None, detector_type="yolo_fine_tunned", conf=0.25, nms=0.45, tsize=(1280, 1280)):
        self.detector_type = detector_type  # "custom" or "pretrained"
        self.conf = conf
        self.nms = nms
        self.tsize = tsize
        
        # Initialize model based on detector type
        if detector_type == "yolo_fine_tunned":
            sys.path.append(os.path.dirname(exp_file))
            exp_name = os.path.basename(exp_file).split(".")[0]
            module = __import__(exp_name)
            self.exp = module.Exp()
        else:  # pretrained
            exp_file = "yolox-l"
            self.exp = get_exp(exp_file=None, exp_name=exp_file)
            
        self.model = self.exp.get_model()
        self.model.eval()
        
        # Class mappings based on detector type
        if detector_type == "yolo_fine_tunned":
            self.class_names = {
                0: 'Pedestrian',
                1: 'Cyclist',
                2: 'Motorbike',
                3: 'Small_motorised_vehicle',
                4: 'Car',
                5: 'Medium_vehicle',
                6: 'Large_vehicle',
                7: 'Bus',
                8: 'Emergency_vehicle'
            }
        else:
            self.class_names = {
                0: 'person',
                1: 'bicycle',
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                6: 'train',
                7: 'truck'
            }
            
        try:
            # Load weights
            ckpt = torch.load(model_path, map_location="cpu")
            if "model" in ckpt:
                model_weights = ckpt["model"]
            elif "state_dict" in ckpt:
                model_weights = ckpt["state_dict"]
            else:
                model_weights = ckpt
            self.model.load_state_dict(model_weights, strict=False)
            
            # Set device
            try:
                if torch.cuda.is_available():
                    self.model.cuda()
                    self.device = "gpu"
                    print("=========Using GPU for inference==============")
                else:
                    self.device = "cpu"
            except Exception as e:
                print(f"CUDA error: {e}")
                self.device = "cpu"
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def inference(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        
        # Validation
        assert height > 0 and width > 0, f"Invalid image dimensions: {width}x{height}"
        r = min(self.tsize[0] / img.shape[0], self.tsize[1] / img.shape[1])
        assert r > 0, f"Invalid resize ratio: {r}"
        
        if self.detector_type == "yolo_fine_tunned":
            # Custom detector preprocessing
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            
            padded_img = np.ones((self.tsize[0], self.tsize[1], 3), dtype=np.float32) * 114.0
            padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
            
            padded_img = padded_img[:, :, ::-1]  # BGR to RGB
            padded_img = padded_img.astype(np.float32) / 255.0
            
            # Apply normalization for custom model
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            padded_img = (padded_img - means) / stds
            
        else:
            # Pretrained model preprocessing
           
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            
            padded_img = np.ones((self.tsize[0], self.tsize[1], 3), dtype=np.float32) * 114.0
            padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
            padded_img = padded_img[:, :, ::-1]  # BGR to RGB
            padded_img = padded_img.astype(np.float32) / 255.0
        
        padded_img = padded_img.transpose(2, 0, 1)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        
        img_info["ratio"] = r
        
        if self.device == "gpu":
            padded_img = torch.from_numpy(padded_img).cuda()
        else:
            padded_img = torch.from_numpy(padded_img)
            
        return padded_img, img_info
    def process_frames(self, frames_dir, gt_dir, matching_threshold=0.5, output_dir=None):
        """Process frames with visualization and evaluation."""
        evaluator = DetectionEvaluator(iou_threshold=matching_threshold)
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        colors = {
            'pedestrian': (125, 125, 255),
            'cyclist': (255, 0, 0),
            'vehicle': (0, 0, 255),
            'motorbike': (255, 255, 0),
            'gt': (0, 255, 0)
        }

        # Collect all frame paths first
        all_frame_paths = []
        for video_folder in sorted(os.listdir(frames_dir)):
            video_frames_path = os.path.join(frames_dir, video_folder)
            v_name = video_folder.split('.')[0]
            
            if not os.path.isdir(video_frames_path):
                continue
            
            gt_file = os.path.join(gt_dir, f"{v_name}.txt")
            if not os.path.exists(gt_file):
                print(f"No ground truth file found for {video_folder}")
                continue
            
            for frame_file in sorted(os.listdir(video_frames_path)):
                if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_frame_paths.append((video_frames_path, frame_file, v_name, gt_file))

        # Wrap the frames with tqdm for progress tracking
        from tqdm.auto import tqdm  # Add this import
        frame_iterator = tqdm(all_frame_paths, desc="Processing Frames", unit="frame")

        # Cache for ground truth detections
        gt_detections_cache = {}
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

        for video_frames_path, frame_file, v_name, gt_file in frame_iterator:
            # Load ground truth detections if not already cached
            if gt_file not in gt_detections_cache:
                gt_detections_cache[gt_file] = self.load_gt_detections(gt_file)
            
            frame_path = os.path.join(video_frames_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
                
            frame_number = int(os.path.splitext(frame_file)[0])
            frame_detections = []
            
            img, img_info = self.inference(frame)
            with torch.no_grad():
                outputs = self.model(img[None, :])
                outputs = self.postprocess(outputs, self.exp.num_classes, self.conf, self.nms)[0]
            
            if outputs is not None and outputs.shape[0] > 0:
                outputs[:, 0:4] /= img_info["ratio"]
                
                if self.detector_type == "yolo_fine_tunned":
                    self._process_custom_detections(outputs, frame, frame_detections, colors)
                else:
                    self._process_pretrained_detections(outputs, frame, frame_detections, colors)
                # print(frame_detections)

            # Draw ground truth boxes and evaluate
            gt_detections = gt_detections_cache[gt_file]
            if frame_number in gt_detections:
                self._draw_ground_truth(frame, gt_detections[frame_number], colors)
            
            frame_gt = gt_detections.get(frame_number, [])
            frame_metrics = evaluator.evaluate_frame(frame_detections, frame_gt)
            evaluator.update_metrics(frame_metrics)

            # Display results
            cv2.imshow('Detection', frame)
            cv2.resizeWindow('Detection', 1280, 720)
            
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
        
        results = evaluator.calculate_final_metrics()
        # print(format_results(results))
        
        cv2.destroyAllWindows()
        return results
    def _process_custom_detections(self, outputs, frame, frame_detections, colors):
        """Process detections for custom model"""
        for det in outputs:
            box = det[:4].cpu().numpy().astype(np.int32)
            score = det[4].cpu().numpy()
            cls_id = int(det[6].cpu().numpy())

            if cls_id == 0:  # Pedestrian
                self._draw_bbox(frame, box, colors['pedestrian'], 'Pedestrian', score)
                frame_detections.append({"bbox": box, "class": "Pedestrian", "score": score})
            elif cls_id == 1:  # Cyclist
                self._draw_bbox(frame, box, colors['cyclist'], 'Cyclist', score)
                frame_detections.append({"bbox": box, "class": "Cyclist", "score": score})
            elif cls_id == 2:  # Motorbike
                self._draw_bbox(frame, box, colors['motorbike'], 'Motorbike', score)
                frame_detections.append({"bbox": box, "class": "Motorbike", "score": score})
            elif cls_id in [3, 4, 5, 6, 7, 8]:  # Vehicle classes
                self._draw_bbox(frame, box, colors['vehicle'], 'Vehicle', score)
                frame_detections.append({"bbox": box, "class": "Vehicle", "score": score})

    def _process_pretrained_detections(self, outputs, frame, frame_detections, colors):
        """Process detections for pretrained model with joint detection logic"""
        # Detection collectors
        detection_groups = {
            'person': [],
            'bicycle': [],
            'motorcycle': [],
            'vehicle': []
        }
        
        # Categorize detections
        for det in outputs:
            box = det[:4].cpu().numpy().astype(np.int32)
            score = det[4].cpu().numpy()
            cls_id = int(det[6].cpu().numpy())
            
            detection_info = {"bbox": box, "score": score}
            
            if cls_id == 0:  # Person
                detection_groups['person'].append(detection_info)
            elif cls_id == 1:  # Bicycle
                detection_groups['bicycle'].append(detection_info)
            elif cls_id == 3:  # Motorcycle
                detection_groups['motorcycle'].append(detection_info)
            elif cls_id in [2, 5, 6, 7]:  # Vehicle classes
                detection_groups['vehicle'].append(detection_info)
        
        # Process all detections (joint and individual) in one go
        frame_detections = self._process_joint_detections(
            frame, 
            frame_detections, 
            detection_groups['person'],
            detection_groups['bicycle'],
            detection_groups['motorcycle'],
            detection_groups['vehicle'],
            colors
        )
        
        
        return frame_detections

    def _create_joint_detection(self, primary_box, secondary_box, primary_score, secondary_score, joint_name):
        """
        Create a joint detection from two bounding boxes.
        
        Args:
            primary_box: First bounding box coordinates
            secondary_box: Second bounding box coordinates
            primary_score: Score of first detection
            secondary_score: Score of second detection
            joint_name: Name for the joint detection class
            
        Returns:
            Dictionary containing joint detection information
        """
        x_min = min(primary_box[0], secondary_box[0])
        y_min = min(primary_box[1], secondary_box[1])
        x_max = max(primary_box[2], secondary_box[2])
        y_max = max(primary_box[3], secondary_box[3])
        
        return {
            'bbox': [x_min, y_min, x_max, y_max],
            'class': joint_name,
            'score': (primary_score + secondary_score) / 2
        }

    def _create_joint_detections(self, primary_list, secondary_list, joint_name, joint_color):
        """
        Creates joint detections between two lists of detections.
        Returns joint detections and indices of used primary/secondary objects.
        """
        joint_dets = []
        used_primary_indices = set()
        used_secondary_indices = set()
        
        for p_idx, primary in enumerate(primary_list):
            for s_idx, secondary in enumerate(secondary_list):
                if s_idx in used_secondary_indices:
                    continue
                    
                iou = self.calculate_iou(primary['bbox'], secondary['bbox'])
                if iou > self.proximity_threshold:
                    joint_det = self._create_joint_detection(
                        primary['bbox'], 
                        secondary['bbox'],
                        primary['score'],
                        secondary['score'],
                        joint_name
                    )
                    joint_dets.append(joint_det)
                    
                    used_primary_indices.add(p_idx)
                    used_secondary_indices.add(s_idx)
                    break
        
        return joint_dets, used_primary_indices, used_secondary_indices

    def _process_joint_detections(self, frame, frame_detections, person_detections, 
                            bicycle_detections, motorcycle_detections, 
                            vehicle_detections, colors, proximity_threshold=0.1):
        """
        Process joint detections for cyclists and motorcycle riders, and handle remaining detections.
        Returns updated frame_detections with all processed detections.
        """
        self.proximity_threshold = proximity_threshold
        used_person_indices = set()
        
        # Process cyclists
        cyclist_detections, used_bicycle_indices, used_person_indices_cyclist = self._create_joint_detections(
            bicycle_detections,
            person_detections,
            'Cyclist',
            colors['cyclist']
        )
        used_person_indices.update(used_person_indices_cyclist)
        
        # Process motorcycle riders
        motorbike_detections, used_motorbike_indices, used_person_indices_motorbike = self._create_joint_detections(
            motorcycle_detections,
            person_detections,
            'Motorbike',
            colors['motorbike']
        )
        used_person_indices.update(used_person_indices_motorbike)
        
        # Draw and add joint detections
        for det_list, color in [(cyclist_detections, colors['cyclist']), 
                            (motorbike_detections, colors['motorbike'])]:
            for det in det_list:
                # self._draw_bbox(frame, 
                #             det['bbox'],
                #             color,
                #             det['class'],
                #             det['score'])
                frame_detections.append(det)
        
        # Handle remaining pedestrians
        for p_idx, person in enumerate(person_detections):
            if p_idx not in used_person_indices:
                # self._draw_bbox(frame, 
                #             person['bbox'], 
                #             colors['pedestrian'], 
                #             'Pedestrian', 
                #             person['score'])
                frame_detections.append({
                    'bbox': person['bbox'],
                    'class': 'Pedestrian',
                    'score': person['score']
                })
        
        # Handle remaining motorcycles
        for m_idx, motorbike in enumerate(motorcycle_detections):
            if m_idx not in used_motorbike_indices:
                # self._draw_bbox(frame, 
                #             motorbike['bbox'], 
                #             colors['motorbike'], 
                #             'Motorbike', 
                #             motorbike['score'])
                frame_detections.append({
                    'bbox': motorbike['bbox'],
                    'class': 'Motorbike',
                    'score': motorbike['score']
                })
        
        # Handle remaining vehicles
        for vehicle in vehicle_detections:
            # self._draw_bbox(frame, 
            #             vehicle['bbox'], 
            #             colors['vehicle'], 
            #             'Vehicle', 
            #             vehicle['score'])
            frame_detections.append({
                'bbox': vehicle['bbox'],
                'class': 'Vehicle',
                'score': vehicle['score']
            })
        
        return frame_detections    
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def _draw_ground_truth(self, frame, gt_detections, colors):
        """Draw ground truth boxes on the frame"""
        for gt_det in gt_detections:
            x1, y1, x2, y2, score, class_id, track_id = gt_det
            
            if class_id in [3, 4, 5, 6, 7, 8]:
                class_label = 'Vehicle'
            else:
                class_label = self.class_names.get(class_id, 'Unknown')
            
            cv2.rectangle(frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        colors['gt'], 
                        2)
            cv2.putText(frame, 
                    f'GT_{class_label}', 
                    (int(x1), int(y1) - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 
                    colors['gt'], 
                    2)

    def _draw_bbox(self, frame, bbox, color, label, score):
        """Draw bounding box with label and score"""
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        label_text = f'{label}: {score:.2f}'
        
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            1
        )
        cv2.rectangle(
            frame, 
            (bbox[0], bbox[1] - text_height - 10), 
            (bbox[0] + text_width, bbox[1]), 
            color, 
            -1
        )
        
        cv2.putText(
            frame, 
            label,#label_text, 
            (bbox[0], bbox[1] - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (125, 125, 255), 
            2
        )

    def load_gt_detections(self, gt_file):
        """Load ground truth detections in KITTI format"""
        print("Loading GT file:", gt_file)
        gt_detections = {}
        
        Gt_Object_Classes = {
            0: 'Pedestrian',
            1: 'Cyclist',
            2: 'Motorbike',
            3: 'Small_motorised_vehicle',
            4: 'Car',
            5: 'Medium_vehicle',
            6: 'Large_vehicle',
            7: 'Bus',
            8: 'Emergency_vehicle'
        }

        class_name_to_id = {name.lower(): id for id, name in Gt_Object_Classes.items()}
        
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 17:
                        frame_id = int(float(parts[0]))
                        track_id = int(float(parts[1]))
                        class_name = parts[2]
                        
                        class_id = class_name_to_id.get(class_name.lower())
                        if class_id is None:
                            continue
                        
                        x1 = float(parts[6])
                        y1 = float(parts[7])
                        x2 = float(parts[8])
                        y2 = float(parts[9])
                        score = float(parts[-1]) if len(parts) > 17 else 1.0
                        
                        if frame_id not in gt_detections:
                            gt_detections[frame_id] = []
                        
                        gt_detections[frame_id].append([x1, y1, x2, y2, score, class_id, track_id])
                        
        except Exception as e:
            print(f"Error loading ground truth file: {e}")
            return {}
        
        return gt_detections


    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        """Postprocess the model output"""
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
 
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                continue
 
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
 
            if not image_pred[conf_mask].size(0):
                continue
 
            detections = torch.cat((
                image_pred[:, :5],
                class_conf,
                class_pred.float(),
                ), 1)[conf_mask]
 
            nms_out_index = torch.ops.torchvision.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
 
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
 
        return output
    


if __name__ == "__main__":
    # Example usage for custom model
    custom_args = {
        'model_path': 'YOLOX_outputs/yolo_emt/latest_ckpt.pth.tar',
        'exp_file': 'ByteTrack/exps/example/mot/yolo_emt.py',
        'frames_dir': "emt/test_frames",
        'gt_dir': "emt/emt_annotations/labels_full",
        'output_dir': "output_custom_eval/",
        'iou_threshold': 0.5,
        'detection_threshold': 0.4,
        'nms_threshold': 0.5
    }

    # Create and run custom detector
    custom_detector = UnifiedVideoDetector(
        model_path=custom_args['model_path'],
        exp_file=custom_args['exp_file'],
        detector_type="yolo_fine_tunned",
        conf=custom_args['detection_threshold'],
        nms=custom_args['nms_threshold'],
        tsize=(1280, 1280)
    )

    custom_results = custom_detector.process_frames(
        frames_dir=custom_args['frames_dir'],
        gt_dir=custom_args['gt_dir'],
        output_dir=custom_args['output_dir'],
        matching_threshold=custom_args['iou_threshold']
    )
    print("\nyolo_fine_tunned Model Results:")
    print(format_results(custom_results))

    # Example usage for pretrained model
    pretrained_args = {
        'model_path': 'pretrained/yolox_l.pth',
        'frames_dir': "emt/test_frames",
        'gt_dir': "emt/emt_annotations/labels_full",
        'output_dir': "output_pretrained_eval/",
        'iou_threshold': 0.5,
        'detection_threshold': 0.4,
        'nms_threshold': 0.5
    }

    # Create and run pretrained detector
    pretrained_detector = UnifiedVideoDetector(
        model_path=pretrained_args['model_path'],
        detector_type="pretrained",
        conf=pretrained_args['detection_threshold'],
        nms=pretrained_args['nms_threshold'],
        tsize=(640, 640)
    )

    pretrained_results = pretrained_detector.process_frames(
        frames_dir=pretrained_args['frames_dir'],
        gt_dir=pretrained_args['gt_dir'],
        output_dir=pretrained_args['output_dir'],
        matching_threshold=pretrained_args['iou_threshold']
    )
    print("\nPretrained Model Results:")
    print(format_results(pretrained_results))