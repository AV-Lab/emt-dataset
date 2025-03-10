### Trajectory Prediction

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