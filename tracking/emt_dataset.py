import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException
from .. import _timing


class EMT2DBox(_BaseDataset):
    """Dataset class for EMT 2D bounding box tracking"""
    @staticmethod
    def get_default_dataset_config():
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/'),
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/'),
            'OUTPUT_FOLDER': None,
            'TRACKERS_TO_EVAL': None,
            'USE_SUPER_CATEGORIES': True,  # New config option
            'CLASSES_TO_EVAL': ['car', 'pedestrian', 'cyclist'],  # Default to super categories
            'SPLIT_TO_EVAL': 'val',
            'INPUT_AS_ZIP': False,
            'PRINT_CONFIG': True,
            'TRACKER_SUB_FOLDER': 'data',
            'OUTPUT_SUB_FOLDER': '',
            'TRACKER_DISPLAY_NAMES': None,
        }
        return default_config


    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.use_super_categories = self.config.get('USE_SUPER_CATEGORIES', False)
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        self.max_occlusion = float('inf')  # No occlusion filtering
        self.max_truncation = float('inf') # No truncation filtering 
        self.min_height = 0         

        # Define valid base classes
        self.valid_classes = ['pedestrian', 'cyclist', 'motorbike', 'small_motorised_vehicle', 
                        'car', 'medium_vehicle', 'large_vehicle', 'bus', 'emergency_vehicle']

        # Base class ID mapping
        self.class_name_to_class_id = {
            'pedestrian': 0,
            'cyclist': 1,
            'motorbike': 2,
            'small_motorised_vehicle': 3,
            'car': 4,
            'medium_vehicle': 5,
            'large_vehicle': 6,
            'bus': 7,
            'emergency_vehicle': 8
        }

        # Define super categories
        self.super_categories = {
            "VEHICLE": ["small_motorised_vehicle", "car", "medium_vehicle", 
                    "large_vehicle", "bus", "emergency_vehicle", "motorbike"],
            "PERSON": ["pedestrian", "cyclist"]
        }

        # Create mapping from base classes to super categories
        self.class_to_super_category = {}
        for super_cat, classes in self.super_categories.items():
            for cls in classes:
                self.class_to_super_category[cls] = super_cat

        # Validate and process classes to evaluate
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                        for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException(f'Attempted to evaluate invalid classes. Valid classes are: {self.valid_classes}')

        # If using super categories, group the evaluation classes
        if self.use_super_categories:
            selected_super_cats = set()
            for cls in self.class_list:
                super_cat = self.class_to_super_category.get(cls)
                if super_cat:
                    selected_super_cats.add(super_cat)
            self.eval_categories = list(selected_super_cats)
            # Create ID mapping for super categories
            self.category_to_id = {cat: idx for idx, cat in enumerate(self.eval_categories)}
        else:
            self.eval_categories = self.class_list
            self.category_to_id = self.class_name_to_class_id

        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        seqmap_name = 'evaluate_tracking.seqmap.' + self.config['SPLIT_TO_EVAL']
        seqmap_file = os.path.join(self.gt_fol, seqmap_name)
        # print(f'os.path.basename(seqmap_file): {os.path.basename(seqmap_file)}')
        if not os.path.isfile(seqmap_file):
            raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))

        # Open the seqmap file and process it
        with open(seqmap_file) as fp:
            for line in fp:
                # Split the line into sequence ID and length
                parts = line.strip().split()
                
                # Ensure the line is valid and has two parts (seq_id and length)
                if len(parts) == 2:
                    seq = parts[0]  # Sequence ID
                    seq_length = int(parts[1])  # Length of the sequence

                    # Add the sequence to the list and set the sequence length
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = seq_length

                    # Check if the ground truth file exists for the sequence
                    if not self.data_is_zipped:
                        curr_file = os.path.join(self.gt_fol, 'labels', 'video_' + seq + '.txt')
                        print(f"\nGT file: \t{curr_file} Found!")

                        # Raise an exception if the ground truth file is not found
                        if not os.path.isfile(curr_file):
                            raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

                    # Handle the case for zipped data (if applicable)
                    if self.data_is_zipped:
                        curr_file = os.path.join(self.gt_fol, 'data.zip')
                        if not os.path.isfile(curr_file):
                            raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))   

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        # Set up tracker display names
        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        # Check all tracker files exist
        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, 'video_' + seq + '.txt')
                    if not os.path.isfile(curr_file):
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))
    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the EMT 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = os.path.join(self.gt_fol, 'labels', 'video_' + seq + '.txt')# os.path.join(self.gt_fol, seq + '.txt')  # Updated path
            else:
                file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol,'video_' + seq + '.txt')

        crowd_ignore_filter = None  # No crowd ignore regions in your dataset
        
        #Valid classes  Change to just 
        valid_filter = {2: [x for x in self.class_list]}  # Just use your class list directly

        # Convert EMT class strings to class ids
        convert_filter = {2: self.class_name_to_class_id}

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, time_col=0, id_col=1, remove_negative_ids=True,
                                                             valid_filter=valid_filter,
                                                             crowd_ignore_filter=crowd_ignore_filter,
                                                             convert_filter=convert_filter,
                                                             is_zipped=self.data_is_zipped, zip_file=zip_file)
        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys

        current_time_keys = [str(t) for t in range(1,num_timesteps+1)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys])) 
        
            

        for t in range(num_timesteps):
            time_key = str(t)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=np.float)
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 6:10])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data['classes'][t] = np.atleast_1d(time_data[:, 2]).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.atleast_1d(time_data[:, 3].astype(int)),
                                      'occlusion': np.atleast_1d(time_data[:, 4].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    if time_data.shape[1] > 17:
                        raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 17])
                    else:
                        raw_data['tracker_confidences'][t] = np.ones(time_data.shape[0])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.empty(0),
                                      'occlusion': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                # if time_key in ignore_data.keys():
                #     time_ignore = np.asarray(ignore_data[time_key], dtype=np.float)
                #     raw_data['gt_crowd_ignore_regions'][t] = np.atleast_2d(time_ignore[:, 6:10])
                # else:
                    raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Preprocess data for a single sequence for evaluation.
        
        Inputs:
            - raw_data: dict containing the sequence data read by get_raw_seq_data()
            - cls: class or super-category to be evaluated
        
        Outputs:
            - data: dict containing evaluation information:
                [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets]: integers
                [gt_ids, tracker_ids, tracker_confidences]: lists of 1D NDArrays
                [gt_dets, tracker_dets]: lists of detection lists
                [similarity_scores]: list of 2D NDArrays
        """
        # Get class IDs to evaluate based on mode
        # Get class IDs to evaluate based on class type
        if self.use_super_categories:
            # Convert base class to its super category if needed
            if cls in self.class_to_super_category:
                super_cat = self.class_to_super_category[cls]
                constituent_classes = self.super_categories[super_cat]
            elif cls in self.super_categories:
                constituent_classes = self.super_categories[cls]
            else:
                raise TrackEvalException(f'Invalid class or super category: {cls}')
            class_ids = [self.class_name_to_class_id[c] for c in constituent_classes]
        else:
            # Normal class evaluation
            if cls not in self.class_name_to_class_id:
                raise TrackEvalException(f'Invalid class {cls}')
            class_ids = [self.class_name_to_class_id[cls]]

        # Initialize data containers
        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 
                    'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        # print("class_ids: {}".format(class_ids))
        # Process each timestep
        for t in range(raw_data['num_timesteps']):
            # Create class masks for ground truth and tracker
            gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in class_ids], axis=0)
            gt_class_mask = gt_class_mask.astype(bool)
            
            tracker_class_mask = np.sum([raw_data['tracker_classes'][t] == c for c in class_ids], axis=0)
            tracker_class_mask = tracker_class_mask.astype(bool)

            # Extract relevant detections
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]
            gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]

            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match detections using Hungarian algorithm
            to_remove_matched = []
            unmatched_indices = np.arange(tracker_ids.shape[0])
            
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                # Check for occlusion and truncation
                is_occluded_or_truncated = np.logical_or(
                    gt_occlusion[match_rows] > self.max_occlusion + np.finfo('float').eps,
                    gt_truncation[match_rows] > self.max_truncation + np.finfo('float').eps)
                
                to_remove_matched = match_cols[is_occluded_or_truncated]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # Handle unmatched detections
            unmatched_tracker_dets = tracker_dets[unmatched_indices]
            unmatched_heights = unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
            is_too_small = unmatched_heights <= self.min_height + np.finfo('float').eps

            # Apply preprocessing filters
            to_remove_unmatched = unmatched_indices[is_too_small]
            # Ensure both arrays exist before concatenation
            if len(to_remove_matched) == 0 and len(to_remove_unmatched) == 0:
                to_remove_tracker = np.array([], dtype=np.int)
            else:
                to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched))

            # to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched), axis=0)
            
            # Update data structures
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Filter ground truth
            gt_to_keep_mask = (np.less_equal(gt_occlusion, self.max_occlusion)) & \
                            (np.less_equal(gt_truncation, self.max_truncation))
            
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            # Update counters
            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Relabel IDs to be contiguous
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record statistics
        data.update({
            'num_tracker_dets': num_tracker_dets,
            'num_gt_dets': num_gt_dets,
            'num_tracker_ids': len(unique_tracker_ids),
            'num_gt_ids': len(unique_gt_ids),
            'num_timesteps': raw_data['num_timesteps'],
            'seq': raw_data['seq']
        })

        # Ensure unique IDs per timestep
        self._check_unique_ids(data)
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores
