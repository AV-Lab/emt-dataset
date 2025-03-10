from detector import UnifiedVideoDetector
import torch
import numpy as np
# import motmetrics as mm
import pandas as pd
import cv2

from tqdm import tqdm
# from tracker_utils import print_args,DetectionSource,TrackerConfig,Trackers,load_gt_detections,CLASSES_OF_INTEREST,save_to_kitti_format,load_detector,get_detections


# Define classes we want to detect
CLASSES_OF_INTEREST = {
   0: 'Pedestrian',
   1: 'Cyclist', 
   2: 'Motorbike',
   3: 'Vehicle',
}

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


def get_detections(frame, frame_counter, detection_type, detector=None, gt_detections=None):
    """Get detections from specified source"""
    if detection_type in [DetectionSource.Yolo_fine_tunned, DetectionSource.Yolo_off_shelf]:
        if detector is None:
            print("detector is None")
            return None, None, None
            
        # Initialize frame_detections as in detector
        frame_detections = []
        
        # Process using detector's methods
        img, img_info = detector.inference(frame)
        with torch.no_grad():
            outputs = detector.model(img[None, :])
            outputs = detector.postprocess(outputs, detector.exp.num_classes, detector.conf, detector.nms)[0]
        
        if outputs is not None and outputs.shape[0] > 0:
            outputs[:, 0:4] /= img_info["ratio"]
            
            # Use detector's processing methods with same colors
            colors = {
                'pedestrian': (125, 125, 255),
                'cyclist': (255, 0, 0),
                'vehicle': (0, 0, 255),
                'motorbike': (255, 255, 0),
                'gt': (0, 255, 0)
            }
            
            if detection_type == DetectionSource.Yolo_fine_tunned:
                detector._process_custom_detections(outputs, frame, frame_detections, colors)
            else:
                detector._process_pretrained_detections(outputs, frame, frame_detections, colors)
                
            if frame_detections:
                # Convert detector's frame_detections to required output format
                class_mapping = {
                    'Pedestrian': 0,
                    'Cyclist': 1,
                    'Motorbike': 2,
                    'Vehicle': 3
                }
                
                dets = np.array([
                    list(det['bbox']) +  # x1,y1,x2,y2
                    [det['score']] +     # score
                    [class_mapping[det['class']]]  # map class name to ID
                    for det in frame_detections
                ])
                class_ids = np.array([class_mapping[det['class']] for det in frame_detections], dtype=np.int32)
                return dets, class_ids, None
            
    elif detection_type == DetectionSource.GT:
        if frame_counter in gt_detections:
            frame_dets = np.array(gt_detections[frame_counter])
            dets = frame_dets[:, :6]  # get x1,y1,x2,y2,score,class_id
            class_ids = frame_dets[:, 5].astype(int)  # get class_id
            track_ids = frame_dets[:, 6].astype(int)  # get track_id
            return dets, class_ids, track_ids

    return None, None, None
def load_detector(detection_type):
    """
    Load detector model, automatically downloading if needed.
    
    Args:
        detection_type (str): Type of detection source from DetectionSource class
        
    Returns:
        UnifiedVideoDetector
    """
    try:
        # print(f"-------------detection_type {detection_type} --------------")
        if detection_type == DetectionSource.Yolo_fine_tunned:
            print("\n\n==================Loading Yolo Fine-tunned Detetector=====================\n\n")
            # Will automatically download YOLOv10x if not present
            fine_tunned_args = {
                'model_path': 'YOLOX_outputs/yolo_emt/latest_ckpt.pth.tar',
                'exp_file': 'ByteTrack/exps/example/mot/yolo_emt.py',
                'frames_dir': "emt/frames",
                'gt_dir': "emt/emt_annotations/labels",
                'output_dir': "output_custom_eval/",
                'iou_threshold': 0.5,
                'detection_threshold': 0.4,
                'nms_threshold': 0.5
            }

            fine_tunned = UnifiedVideoDetector(
                model_path=fine_tunned_args['model_path'],
                exp_file=fine_tunned_args['exp_file'],
                detector_type="yolo_fine_tunned",
                conf=fine_tunned_args['detection_threshold'],
                nms=fine_tunned_args['nms_threshold'],
                tsize=(1280, 1280)
            )
            return fine_tunned
        elif detection_type == DetectionSource.Yolo_off_shelf:
            print("\n\n==================Loading Yolo Off-shelf Detector=====================\n\n")
            pretrained_args = {
                'model_path': 'pretrained/yolox_l.pth',
                'frames_dir': "emt/frames",
                'gt_dir': "emt/emt_annotations/labels",
                'output_dir': "output_pretrained_eval/",
                'iou_threshold': 0.5,
                'detection_threshold': 0.3,
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
            return pretrained_detector
        return None
        
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        return None
def save_to_kitti_format(detections, frame_id, output_file, frame_offset=0):
    """Save detections in KITTI format efficiently
    KITTI format per line:
    frame_id track_id object_class truncation occlusion alpha 
    bbox_left bbox_top bbox_right bbox_bottom height width length 
    x y z rotation_y score
    """
    # Prepare default values for 3D fields
    defaults = {
        'truncation': -1,
        'occlusion': -1,
        'alpha': -1,
        'height': -1,
        'width': -1,
        'length': -1,
        'x': -1,
        'y': -1,
        'z': -1,
        'rotation_y': -1
    }
    
    # Build all lines at once
    lines = []
    for box, track_id, class_id, conf in zip(detections.xyxy, 
                                           detections.tracker_id, 
                                           detections.class_id, 
                                           detections.confidence):
        if class_id not in CLASSES_OF_INTEREST:
            continue
            
        class_name = CLASSES_OF_INTEREST[class_id]
        x1, y1, x2, y2 = box
        
        # Format line with all fields
        line = f"{frame_id} {track_id} {class_name} "
        line += f"{defaults['truncation']} {defaults['occlusion']} {defaults['alpha']} "
        line += f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} "
        line += f"{defaults['height']:.2f} {defaults['width']:.2f} {defaults['length']:.2f} "
        line += f"{defaults['x']:.2f} {defaults['y']:.2f} {defaults['z']:.2f} "
        line += f"{defaults['rotation_y']:.2f} {conf:.2f}\n"
        
        lines.append(line)
    
    # Write all lines at once
    if lines:
        output_file.writelines(lines)
class DetectionSource:
    """Class defining available detection sources and their configurations."""
    
    # Detection sources
    Yolo_fine_tunned = "Yolo_fine_tunned"
    Yolo_off_shelf = "Yolo_off_shelf"
    GT = "gt"
    
    # Detector names/versions
    DETECTOR_NAMES = {
        Yolo_fine_tunned : "Yolo_fine_tunned",
        Yolo_off_shelf : "Yolo_off_shelf",
        GT: "Ground Truth"
    }
    
    @classmethod
    def get_detector_name(cls, source):
        """Get the full name of the detector for a given source.
        
        Args:
            source (str): Detection source identifier (e.g., 'yolo', 'gt')
            
        Returns:
            str: Full name of the detector
        """
        return cls.DETECTOR_NAMES.get(source, "Unknown Detector")
def print_args(args, verbose=False):
    """
    Print argument values in either verbose or simple format
    
    Args:
        args: Parsed argument namespace from argparse
        verbose: If True, print detailed output with descriptions
    """
    arg_dict = vars(args)
    
    if verbose:
        print("\n" + "="*50)
        print("CONFIGURATION SETTINGS")
        print("="*50)
        
        descriptions = {
            'input': 'Input folder path (videos/frames)',
            'output': 'Output folder path',
            'gt_folder': 'Ground truth annotations folder',
            'tracker_name': 'Tracker to be used',
            'use_frames': 'Using frame folders as input',
            'det': 'Detection source (YOLO/GT)',
            'use_gt_tracks': 'Using GT for Tracking',
            'show_display': 'Visualization enabled',
            'save_video': 'Output video saving enabled',
            'batch_size': 'Processing batch size',
            'verbose': 'Verbose output mode'
        }
        
        max_arg_len = max(len(arg) for arg in arg_dict.keys())
        max_val_len = max(len(str(val)) for val in arg_dict.values())
        
        for arg, value in arg_dict.items():
            description = descriptions.get(arg, 'No description available')
            print(f"{arg:<{max_arg_len}} : {str(value):<{max_val_len}} | {description}")
        
        print("="*50 + "\n")
    else:
        essential_args = ['input', 'output', 'det', 'batch_size']
        bool_flags = ['use_frames', 'gt_gt', 'show_display', 'save_video', 'verbose']
        
        parts = [f"{arg}={arg_dict[arg]}" for arg in essential_args]
        enabled_flags = [arg for arg in bool_flags if arg_dict[arg]]
        
        if enabled_flags:
            parts.extend(enabled_flags)
            
        print("\nConfig:", ", ".join(parts) + "\n")

class TrackerConfig:
    """Configuration for different trackers"""
    def __init__(self, 
                 track_thresh=0.25,
                 track_buffer=30,
                 match_thresh=0.8,
                 aspect_ratio_thresh=3.0,
                 mot20=False,
                 track_high_thresh=0.6,    # For BOT/BOOST
                 track_low_thresh=0.1,     # For BOT/BOOST
                 new_track_thresh=0.7,     # For BOT/BOOST
                 fps=10):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.mot20 = mot20
        # Additional parameters for other trackers
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.fps = fps

    @classmethod
    def byte_tracker_config(cls):
        """Default configuration for ByteTracker"""
        return cls(
            track_thresh=0.5,
            track_buffer=10,
            match_thresh=0.8
        )

    @classmethod
    def bot_tracker_config(cls):
        """Default configuration for BOT-SORT"""
        return cls(
            track_thresh=0.5,
            track_buffer=10,
            match_thresh=0.8,
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            # proximity_thresh=0.5,
            # appearance_thresh=0.25
        )

    @classmethod
    def boost_tracker_config(cls):
        """Default configuration for BoostTracker"""
        return cls(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.7,
            new_track_thresh=0.7
        )
    


class Trackers:
    """Available tracker types and initialization methods"""
    byte_tracker = "byte"
    bot_tracker = "bot" 
    boost_tracker = "boost"

    _folder_names = {
        byte_tracker: "ByteTracker",
        bot_tracker: "BOT-SORT",
        boost_tracker: "BoostTracker"
    }

    def __init__(self, tracker_name, fps=10):
        if tracker_name not in [self.byte_tracker, self.bot_tracker, self.boost_tracker]:
            raise ValueError(f"Unknown tracker: {tracker_name}. Use 'byte', 'bot', or 'boost'")
        self.tracker_name = tracker_name
        self.fps = fps
        self.config = self.get_config(self.tracker_name)
        self.folder_name = self._folder_names[self.tracker_name]

    def get_config(self, tracker_name):
        config_map = {
            self.byte_tracker: TrackerConfig.byte_tracker_config(),
            self.bot_tracker: TrackerConfig.bot_tracker_config(),
            self.boost_tracker: TrackerConfig.boost_tracker_config()
        }
        return config_map[tracker_name]
   
    @staticmethod 
    def get_tracker_folder(tracker_name, use_gt_tracks=False):
        if use_gt_tracks:
            return 'GtTracker'
        return Trackers._folder_names[tracker_name]

    def get_tracker(self, fps=None, config=None):
        if fps is None:
            fps = self.fps
        if config is not None:
            self.config = config
        self.config.fps = fps

        if self.tracker_name == self.byte_tracker:
            from yolox.tracker.byte_tracker import BYTETracker
            return BYTETracker(self.config, fps)
        elif self.tracker_name == self.bot_tracker:
            from tracker.bot_sort import BoTSORT
            return BoTSORT(self.config, fps)
        elif self.tracker_name == self.boost_tracker:
            raise NotImplementedError("BOOST tracker not implemented")
        else:
            raise ValueError(f"Unknown tracker: {self.tracker_name}")
        


def load_gt_detections(gt_file):
        """Load ground truth detections in KITTI format"""
        print("Loading GT file:", gt_file)
        gt_detections = {}
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
                        if class_id in [4,5,6,7,8]:
                            class_id = 3
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


def handle_visualization(frame, detections, detection_type, box_annotator, writer, show_display,label_annotator=None):
    """Handle frame visualization and video writing with optimized performance"""
    if not (writer is not None or show_display):
        return False  # Skip if no visualization needed
    
    # Only copy frame once if needed
    annotated_frame = frame.copy() if (detections is not None and (writer is not None or show_display)) else frame
    
    if detections is not None:
        # Prepare labels only if needed
        if show_display or writer is not None:
            if detection_type == DetectionSource.GT:
                labels = [f"#{t_id} {CLASSES_OF_INTEREST[int(c_id)]}"
                         for t_id, c_id in zip(detections.tracker_id, detections.class_id)]
            else:
                labels = [f"#{t_id} {CLASSES_OF_INTEREST[int(c_id)]}"
                         for t_id, c_id in zip(detections.tracker_id, detections.class_id)]

            # Annotate frame in-place
            box_annotator.annotate(annotated_frame, detections=detections)  # Add the labels parameter)
            if label_annotator:
                label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    # Handle display and video writing
    if writer is not None:
        writer.write(annotated_frame)
        
    if show_display:
        # Resize for display if frame is too large
        display_frame = annotated_frame
        if annotated_frame.shape[1] > 1280:  # If width > 1280
            scale = 1280 / annotated_frame.shape[1]
            display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
        
        cv2.imshow('Detections', display_frame)
        key = cv2.waitKey(1)
        return key & 0xFF == ord('q')
    
    return False