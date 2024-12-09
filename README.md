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
| Bounding Boxes | 277,150 |
| Unique Agents | 4,209 | 
| Vehicle Instances | 3,729 |
| Pedestrian Instances | 222 | 

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

### 🔄 Format Conversion
To convert annotations to KITTI or GMOT format:
```bash
python kitti_gmot.py
```

### 📊 Data Visualization

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
- Annotations are available for every third frame to optimize the balance between annotation density and processing efficiency

## 📧 Contact
- 👩‍💻 Nadya: 100049370@ku.ac.ae
- 👨‍💻 Murad: murad.mebrahtu@ku.ac.ae