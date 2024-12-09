# EMT dataset 
## Introduction
ROAD-UAE is a richly annotated dataset containing detailed semantic annotations for road agents and events. With 57 minutes of continuous footage, each video segment lasts between 2.5 to 3 minutes. The dataset includes heterogeneous annotations for three main agent categories: people (pedestrians and cyclists), vehicles (divided into seven classes), and traffic lights.

It captures the unique road topology and traffic conditions of the Gulf Region, offering consistent tracking IDs for all road agents. Additionally, it provides action annotations as "action tubes" and road event detections labeled as triplets (Agent, Action, Location) at the frame level.


## 📊 EMT Dataset Features

### Data Collection
| Aspect | Description |
|:-------|:------------|
| Duration | 57 minutes total footage |
| Segments | 2.5-3 minutes continuous recordings |
| Format | 1920x1080, 30fps |
| Storage | Uncompressed raw data |
### Annotation Types
1. **👥 Agent Categories**   
Two people classes and seven vehicle classes.
    - 👤 People  
        * Pedestrians
        * Cyclists
    - 🚗 Vehicles 
        * Motorbike
        * Small motorised vehicle
        * Medium vehicle
        * Large vehicle
        * Car
        * Bus
        * Emergency vehicle   

2. **🎯 Tracking System**  
    - Bounding box annotations
    - Consistent tracking IDs



### Stattistics 
| Category | Count |
|:---------|:--------|
| Annotated Frames | 34,386 | 
| Bounding Boxes | 277,150 |
| Unique Agents | 4,209 | 
| Vehicle Instances | 3,729 |
| Pedestrian Instances | 222 | 


### Dataset Structure 
```
emt/
├── raw/
│   ├── videos/
└── annotations/
|   ├── video_annotations/
├── kitti_annotations/
│   ├── video_annotations/
├── gmot_annotations/
│   ├── video_annotations/
```
# Quick Start Guide

## Plotting Frame Annotations
To generate annotated visualizations from the frames, run:
```bash
python plot_annotations.py frames_dir annotations_dir
```

The script provides two visualization options:
1. Plots all annotations on frames with bounding boxes and labels
2. Generates separate videos for each tracked agent (uncomment `plot_each_agent` in the script)

### Output
- Creates an `output` directory inside your annotations directory
- Saves annotated video as `annotated_video.mp4`
- Saves individual annotated frames in `frames` subdirectory
- When using `plot_each_agent`, saves individual videos for each tracked object

### Visualization Format
- Bounding boxes are drawn around detected objects
- Labels show (agent, action, landmark) for each object
- Frame-by-frame visualization is saved both as images and video

## Visualizing Video Annotations
To visualize annotations overlaid on the raw video, use one of the following commands:

```bash
# Basic visualization with default annotation path
python visualize.py --video_path path/to/video.mp4

# Specify custom annotation path
python visualize.py --video_path path/to/video.mp4 --annotation_path path/to/annotations.txt

# Save the annotated video
python visualize.py --video_path path/to/video.mp4 -save
```

The `-save` flag is optional:
- With `-save`: Creates an annotated video file in `emt/annotated_videos/`
- Without `-save`: Displays the visualization without saving

By default, the script looks for annotation files in `emt/kitti_annotations/` with the same name as the video file.

## Kitti and Gmot annotations
To generate kitti or gmot formatted annotations from the original annotations; use the following command:
```bash
python kitti_gmot.py
```

## Important Note
The annotation data is available for every third frame in the video sequence. This sampling rate was chosen to balance annotation density with processing efficiency.