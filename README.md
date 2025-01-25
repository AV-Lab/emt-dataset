# EMT Dataset

<p align="center">
    <img src="assets/Multi_object_tracking_with _gt.gif" width="1200px"/>
    <br>
    <i>Multi-object tracking with ground truth annotations</i>
</p>

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Quick Start](#quick-start)
- [Analysis Tools](#analysis-tools)
- [Dataset Structure](#dataset-structure)
- [Links](#links)
- [Contact](#contact)

## Introduction
EMT is a comprehensive dataset for autonomous driving research, containing 57 minutes of diverse urban traffic footage from the Gulf Region. The dataset provides rich semantic annotations across three agent categories: people (pedestrians and cyclists), vehicles (seven classes), and traffic lights. Each video segment spans 2.5-3 minutes, capturing challenging real-world scenarios:

- **Dense Urban Traffic**: Complex multi-agent interactions in congested scenarios
- **Weather Variations**: Clear and rainy conditions
- **Visual Challenges**: High reflections from road surfaces and adverse weather combinations (rainy nights)

The dataset provides dense annotations (every third frame) for:
- **Detection & Tracking**: Multi-object tracking with consistent IDs
- **Trajectory Prediction**: Future motion paths and social interactions
- **Intention Prediction**: Behavior understanding in complex scenarios

Validated through benchmarking on state-of-the-art models across tracking, trajectory prediction, and intention prediction tasks, with corresponding ground truth annotations for each benchmark.


### Data Collection
| Aspect | Description |
|:-------|:------------|
| Duration | 57 minutes total footage |
| Segments | 2.5-3 minutes continuous recordings |
| FPS | 10fps for annotated frames |
| Agent Classes | 2 Person classes and 7 Vehicle classes|

### Dataset Statistics
| Category | Count |
|----------|------------|
| Annotated Frames | 34,386 |
| Bounding Boxes | 626,634 |
| Unique Agents | 9,094 |
| Vehicle Instances | 7,857 |
| Pedestrian Instances | 568 |

### Agent Categories
1. **People**   
   - Pedestrians
   - Cyclists

2. **Vehicles**
   - Motorbike
   - Small motorised vehicle
   - Medium vehicle
   - Large vehicle
   - Car
   - Bus
   - Emergency vehicle

## Quick Start

1. Download dataset:
```bash
chmod +x download.sh
./download.sh
```

2. Convert formats:
```bash
python kitti_gmot.py
```

## Analysis Tools

```bash
# Statistics
python stats.py annot_dir

# Visualization
python visualize.py --video_path <path> [--annotation_path <path>] [-save]

# COCO Format Conversion
python coco.py emt/
```

## Dataset Structure

```
emt-dataset/
├── data/
│   ├── annotations/
│   │   ├── intention_annotations/    # Agent intention labels
│   │   ├── tracking_annotations/     # Multi-object tracking data
│   │   ├── prediction_annotations/   # Behavior prediction labels
│   │   └── metadata.txt             # Dataset metadata
│   └── videos/                      # Raw video sequences
```

## Links
- Repository: [GitHub - AV-Lab/road-uae](https://github.com/AV-Lab/road-uae)
- Website: [EMT Dataset](https://avlab.io/emt-dataset/)

## Contact
- Nadya: 100049370@ku.ac.ae
- Murad: murad.mebrahtu@ku.ac.ae
