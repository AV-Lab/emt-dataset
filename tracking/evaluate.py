import os
import trackeval
import sys
from io import StringIO

# Paths
EMT_GT_PATH = 'emt/emt_annotations/'  # Ground truth folder
EMT_TRACKERS_PATH = 'Trackers/ByteTracker/'#'GtTracker/'#'Trackers/ByteTracker/'#'Trackers/GtTracker/'#'Trackers/ByteTracker/' #'Trackers/BOT-SORT/' #'Trackers/ByteTracker/' #'Trackers/GtTracker/'  # Tracker results folder

# Tracker and dataset details
TRACKER_NAME = 'Yolo_off_shelf'#'Yolo_off_shelf'#'Yolo_fine_tunned'#'Ground Truth' #'YOLOv11x'#'ground_truth'  # Name of your tracker
TRACKER_SUB_FOLDER = 'tracked_predictions'#'gt'  # Leave empty if results are in the default folder

class TeeStream:
    """Stream wrapper that writes to multiple streams simultaneously"""
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

def save_and_print_eval(dataset_list, metrics_list, evaluator, output_file):
    """
    Captures evaluation output, saving to file while also printing
    """
    # Setup output capturing
    stdout = sys.stdout
    with open(output_file, 'w') as f:
        # Create TeeStream to write to both file and stdout
        tee = TeeStream(stdout, f)
        sys.stdout = tee
        
        try:
            # Run evaluation
            evaluator.evaluate(dataset_list, metrics_list)
        finally:
            # Restore stdout
            sys.stdout = stdout
            
    print(f"\nResults saved to {output_file}")




# Configurations
def get_emt_config():
    """Configuration for EMT evaluation."""
    return {
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 20,
        'PLOT_CURVES': True,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_CONFIG': True,
        'LOG_ON_ERROR': os.path.join(os.getcwd(), 'error_log.txt'),
        'GT_FOLDER': EMT_GT_PATH,
        'TRACKERS_FOLDER': EMT_TRACKERS_PATH,
        'TRACKER_SUB_FOLDER': TRACKER_SUB_FOLDER,
        'OUTPUT_FOLDER': os.path.join(os.getcwd(), 'EVAL_RESULTS'),
        'TRACKERS_TO_EVAL': [TRACKER_NAME],
        'CLASSES_TO_EVAL':[ 'Pedestrian',  
                   'Vehicle','Cyclist','Motorbike'], #,'Small_motorised_vehicle','Medium_vehicle', 'Large_vehicle', 'Bus', 'Emergency_vehicle'],#  ['car', 'pedestrian'],#  # Select classes to evaluate
        'USE_SUPER_CATEGORIES': False,
        'BENCHMARK': 'kitti',  # KITTI benchmark
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],  # Metrics to evaluate
        'TRACKER_DISPLAY_NAMES': [TRACKER_NAME]
    }


if __name__ == '__main__':
    config = get_emt_config()
    trackers_path,tracker_name = EMT_TRACKERS_PATH.split('/')[:2]
    output_file = trackers_path+'/'+tracker_name+'_'+TRACKER_NAME+f'_results.txt'

    dataset_list = [trackeval.datasets.EMT2DBox(config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    
    evaluator = trackeval.Evaluator(config)
    save_and_print_eval(dataset_list, metrics_list, evaluator, output_file)
