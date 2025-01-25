# 🛣️ EMT Dataset 

## 📑 Table of Contents
- [Introduction](#introduction)
- [Dataset Features](#dataset-features)
- [Dataset Structure](#dataset-structure)
- [Dataset Access](#dataset-access)
- [Usage Guide](#usage-guide)
- [Contact](#contact)

## 🎯 Introduction
EMT is a richly annotated dataset containing detailed semantic annotations for road agents and events. With 57 minutes of continuous footage, each video segment lasts between 2.5 to 3 minutes. The dataset includes heterogeneous annotations for three main agent categories: people (pedestrians and cyclists), vehicles (divided into seven classes), and traffic lights.

It captures the unique road topology and traffic conditions of the Gulf Region, offering consistent tracking IDs for all road agents. Additionally, it provides action annotations as "action tubes" and road event detections labeled as triplets (Agent, Action, Location) at the frame level.

## 📊 Dataset Features

### 📹 Data Collection
| Aspect | Description |
|:-------|:------------|
| Duration | 57 minutes total footage |
| Segments | 2.5-3 minutes continuous recordings |
| Format | 1920x1080, 30fps |
| Storage | Uncompressed raw data |

### 🎯 Agent Categories
1. **👥 People**   
   - 🚶 Pedestrians
   - 🚴 Cyclists

2. **🚗 Vehicles**
   - 🏍️ Motorbike
   - 🚗 Small motorised vehicle
   - 🚐 Medium vehicle
   - 🚛 Large vehicle
   - 🚙 Car
   - 🚌 Bus
   - 🚑 Emergency vehicle

### 📈 Statistics 
| Category | Count |
|:---------|:--------|
| Annotated Frames | 34,386 | 
| Bounding Boxes | 626,634 |
| Unique Agents | 9,094 | 
| Vehicle Instances | 7,857 |
| Pedestrian Instances | 568 | 

## 📁 Dataset Structure 
```
emt/
├── raw/                    # Raw video files
│   └── videos/
├── annotations/            # Original annotations
│   └── video_annotations/
├── kitti_annotations/      # KITTI format annotations
│   └── video_annotations/
└── gmot_annotations/       # GMOT format annotations
    └── video_annotations/
```

## 🔗 Dataset Access
- **💻 Repository**: [GitHub - AV-Lab/road-uae](https://github.com/AV-Lab/road-uae)
- **🌐 Website**: [EMT Dataset](https://avlab.io/emt-dataset/)
- **📦 Download**: [Dropbox Link](https://www.dropbox.com/scl/fo/7f6ww69yzf6ezyj4p9wcp/APUqsrDTct9eoyl0kOvLL8s?rlkey=pvqc2c6vgpzgxjb24x4c3lypy&st=osz5j0ou&dl=0)

## 🚀 Usage Guide
Annotations are available for every third frame to optimize the balance between annotation density and processing efficiency
### 🔄 Format Conversion
To convert annotations to KITTI or GMOT format:
```bash
python kitti_gmot.py
```
### Download Datatset
Download Datat: 
```
wget https://www.dropbox.com/scl/fi/hdcsmc7l688427k5dvslk/annotations.zip?rlkey=nh7gh6t16980nt82kd61ad2lz -O annotations.zip
unzip annotations.zip
```
### 📊 Data Visualization
[Previous sections remain the same until before Usage Guide]

## 🚀 Usage Guide

### 📊 Computing Dataset Statistics
To compute basic statistics about the dataset:
```bash
python stats.py annot_dir
```

This script calculates:
- Total number of bounding box annotations
- Total number of unique agents
- Total number of vehicles
- Total number of pedestrians

Example usage:
```bash
python stats.py emt/kitti_annotations/
```

Sample output:
```
Processing annotation files: ['video_204647.txt', 'video_214547.txt', 'video_142911.txt', 'video_143739.txt', 'video_122233.txt', 'video_204347.txt', 'video_151901.txt', 'video_210906.txt', 'video_115533.txt', 'video_160325.txt', 'video_131333.txt', 'video_054907.txt', 'video_174340.txt', 'video_155425.txt', 'video_054604.txt', 'video_160902-161205.txt', 'video_115833.txt', 'video_125233.txt', 'video_220047.txt', 'video_141432-141733.txt']

-------------------------
Total number of bounding box annotations: 576416
Total number of people (pedestrians and Cyclists): 582
Total number of vehicles: 8234
Total number of unique agents: 8816

-------------------------
Detailed Statistics:
Number of AV agents: 7
Number of unclassified objects: 3
Number of annotation files processed: 20

```

[Rest of Usage Guide sections remain the same]

### 📊 Visualization
#### 1. 🖼️ Frame-by-Frame Visualization
```bash
python plot_annotations.py frames_dir annotations_dir
```

This script offers two visualization modes:
- 🎥 All annotations: Creates a video with bounding boxes and labels for all objects
- 🎯 Individual tracking: Generates separate videos for each tracked agent

**📂 Output:**
- Creates `output` directory inside annotations folder
- Generates:
  - 📼 `annotated_video.mp4`: Complete visualization
  - 📁 `frames/`: Individual annotated frames
  - 🎥 Individual agent videos (when using `plot_each_agent`)

#### 2. 🎥 Video Annotation Visualization
```bash
# Basic visualization
python visualize.py --video_path path/to/video.mp4

# Custom annotation path
python visualize.py --video_path path/to/video.mp4 --annotation_path path/to/annotations.txt

# Save visualization
python visualize.py --video_path path/to/video.mp4 -save
```
**📝 Notes:**
- `-save` flag creates output in `emt/annotated_videos/`
- Default annotation path: `emt/kitti_annotations/<video_name>.txt`


### 📊 Frame extraction and Coco Format

To extract framesfrom kitti_format annotation use the following:
```bash
python coco.py annot_dir
```

This script:
- Extracts annotated frames from kitti_format
- creates coco format annotations
- takes as input the whole dataset where the videos are under raw folder and anotations are under kitti_annotation folder

Example usage:
```bash
python coco.py emt/
```


## 📧 Contact
- 👩‍💻 Nadya: 100049370@ku.ac.ae
- 👨‍💻 Murad: murad.mebrahtu@ku.ac.ae
