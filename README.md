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
EMT is a richly annotated dataset containing detailed semantic annotations for road agents and events. With 57 minutes of continuous footage, each video segment lasts between 2.5 to 3 minutes. The dataset includes heterogeneous annotations for three main agent categories: people (pedestrians and cyclists), vehicles (divided into seven classes), and traffic lights.
 
It captures the unique road topology and traffic conditions of the Gulf Region, offering consistent tracking IDs for all road agents. Additionally, it provides action annotations as "action tubes" and road event detections labeled as triplets (Agent, Action, Location) at the frame level.

## Features

### Data Collection
| Aspect | Description |
|:-------|:------------|
| Duration | 57 minutes total footage |
| Segments | 2.5-3 minutes continuous recordings |
| Format | 1920x1080, 30fps |
| Storage | Uncompressed raw data |

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