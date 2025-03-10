# EMT Dataset

<p align="center">
    <img src="assets/Multi_object_tracking_with _ground-truth.gif" width="1200px"/>
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
EMT is a comprehensive dataset for autonomous driving research, containing 57 minutes of diverse urban traffic footage from the Gulf Region. The dataset provides rich semantic annotations across two agent categories: people (pedestrians and cyclists), vehicles (seven classes). Each video segment spans 2.5-3 minutes, capturing challenging real-world scenarios:

- **Dense Urban Traffic**: Complex multi-agent interactions in congested scenarios
- **Weather Variations**: Clear and rainy conditions
- **Visual Challenges**: High reflections from road surfaces and adverse weather combinations (rainy nights)

The dataset provides dense annotations for:
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

### Dataset Statistics
| Category | Count |
|----------|------------|
| Annotated Frames | 34,386 |
| Bounding Boxes | 626,634 |
| Unique Agents | 9,094 |
| Vehicle Instances | 7,857 |
| Pedestrian Instances | 568 |

| **Class** | **Description** | **Number of Bounding Boxes** | **Number of Agents** |
|-----------|----------------|------------------------------|----------------------|
| Pedestrian | An individual walking on foot. | 24,574 | 568 |
| Cyclist | Any bicycle or electric bike rider. | 594 | 14 |
| Motorbike | Includes motorcycles, bikes, and scooters with two or three wheels. | 11,294 | 159 |
| Car | Any standard automobile. | 429,705 | 6,559 |
| Small motorized vehicle | Motorized transport smaller than a car, such as mobility scooters and quad bikes. | 767 | 13 |
| Medium vehicle | Includes vehicles larger than a standard car, such as vans or tractors. | 51,257 | 741 |
| Large vehicle | Refers to vehicles larger than vans, such as lorries, typically with six or more wheels. | 37,757 | 579 |
| Bus | Covers all types of buses, including school buses, single-deck, double-deck. | 19,244 | 200 |
| Emergency vehicle | Emergency response units like ambulances, police cars, and fire trucks, distinguished by red and blue flashing lights. | 1,182 | 9 |
| **_Overall:_** | | **576,374** | **8,842** |



## Quick Start

1. Download dataset:
```bash
chmod +x download.sh
./download.sh
```

2. To confirm data stats run the following command:
```bash
# Statistics
python dataset_statistics.py
```
3. To visualize detection data run the following command:
```bash
# visualize
cd visualize
python tracking_visualize.py
```
## Dataset Structure
To use base models, the dataset structure should be as follows:
```
emt-dataset/
├── data/
│   ├── annotations/
│   │   ├── intention_annotations/    # Agent intention labels
│   │   ├── tracking_annotations/     # Multi-object tracking data
│   │   ├── prediction_annotations/   # Behavior prediction labels
│   │   └── metadata.txt             # Dataset metadata
│   │── frames/                      # extracted frames(annotated frames only)
│   └── videos/                      # Raw video sequences
```
NB: Videos are not necessary unless you intend to use visual cues 

## Benchmarking
We benchmark the dataset for the following tasks:
- **Multi-class MOT Tracking**: Multi-object tracking with consistent IDs
- **Trajectory Prediction**: Future motion paths and social interactions
- **Intention Prediction**: Behavior understanding in complex scenarios

### Multi-class MOT Tracking

### Trajectory Prediction
This script runs trajectory prediction models using LSTM, Graph Neural Networks (GNNs), and Transformer-based architectures. It supports training and evaluation modes, allowing users to load pre-trained models and process trajectory data with different settings.

```
├── Prediction/
│   ├── dataloaders/
│   ├── evaluation/                      
│   ├── models/
│   │   ├── gat_temporal.py  
│   │   ├── gat.py  
│   │   ├── gcn_temporal.py  
│   │   ├── gcn.py  
│   │   ├── rnn.py  
│   │   ├── transformer_GMM.py  
│   │   └── transformer.py
│   ├── results/  
|   └── run.py
```

To train or evaluate the trajectory prediction model, run the script using the command:
```bash
python run.py <past_trajectory> <future_trajectory> [OPTIONS]
```

#### Arguments

##### Required Arguments:
- **`past_trajectory` (int):** Number of past timesteps used as input for prediction.
- **`future_trajectory` (int):** Number of timesteps into the future to predict.

##### Optional Arguments:
- **`--window_size` (int, default=1):** Sliding window size for processing trajectory data.
- **`--max_nodes` (int, default=50):** Maximum number of nodes used in the Graph Neural Network (GNN) model.
- **`--predictor` (str, default='transformer-gmm'):** Type of predictor model to use. Options include:
  - `lstm`
  - `gcn`
  - `gcn_lstm`
  - `gat`
  - `gat_lstm`
  - `transformer`
  - `transformer-gmm`
- **`--setting` (str, default='train'):** Execution mode:
  - `train`: Train the model
  - `evaluate`: Evaluate a trained model
- **`--checkpoint` (str, default=None):** Path to a model checkpoint file (required for evaluation).
- **`--annotations_path` (str, optional):** Path to annotations if stored in a non-default location.
- **`--num_workers` (int, default=8):** Number of workers used for data loading.
- **`--normalize` (bool, default=False):** Whether to normalize data (recommended: `True`).
- **`--batch_size` (int, default=64):** Batch size for training/evaluation.
- **`--device` (str, default='cuda:0'):** Device to run the model (`cuda:0`, `cuda:1`, or `cpu`).
- **`--seed` (int, default=42):** Random seed for reproducibility (set to 0 for random seed generation).

#### Examples
##### Train a model:
```bash
python run.py 10 20 --predictor transformer --setting train --batch_size 32
```

##### Evaluate a model:
```bash
python run.py 10 20 --setting evaluate --checkpoint path/to/checkpoint.pth
```

#### Installation Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

#### Notes
- Ensure GPU support (`cuda`) is available if running on a GPU.
- When evaluating a model, a valid checkpoint file must be specified.
- Normalization is recommended for better model performance.



## 🔗 Links
- Repository: [GitHub - AV-Lab/road-uae](https://github.com/AV-Lab/road-uae)
- Website: [EMT Dataset](https://avlab.io/emt-dataset/)

## 📝 Citation
If you use the EMT dataset in your research, please cite our paper:

```
@article{EMTdataset2025,
      title={EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region}, 
      author={Nadya Abdel Madjid and Murad Mebrahtu and Abdelmoamen Nasser and Bilal Hassan and Naoufel Werghi and Jorge Dias and Majid Khonji},
      year={2025},
      eprint={2502.19260},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.19260}, 
}
```
## Contact
- Murad: murad.mebrahtu@ku.ac.ae
- Nadya: 100049370@ku.ac.ae
