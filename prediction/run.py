#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Sat Sep 28 18:08:42 2024

# @author: nadya started it but murad messed it up
# """

# import os
# import cv2
# import sys
# import argparse
# import numpy as np
# import json    
# from torch.utils.data import DataLoader
# import torch.cuda as cuda

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils import load_meta_data, set_seeds
# from utils import generate_prediction_settings
# from utils import generate_intention_settings

# from evaluation.distance_metrics import calculate_ade, calculate_fde
# from dataloaders.seq_loader import SeqDataset
# # from dataloaders.frame_loader import GNNDataset
# # from models.rnn import RNNPredictor
# # from models.gnn import GCNPredictor, GATPredictor
# # from models.transformer_model import TransformerPredictor
# from models.transformer import AttentionEMT
# from models.tranformer_GMM import AttentionGMM
# import torch
# import torch.nn as nn


# if __name__ == '__main__':
#     p = argparse.ArgumentParser(description='Run predictors')
#     p.add_argument('past_trajectory',type=int, help='Past Trajectory')
#     p.add_argument('future_trajectory',type=int, help='Prediction Horizon')
#     p.add_argument('epochs', type=str, default=100, help='Num of training epochs')
#     p.add_argument('--window_size', default=1, type=int, help='Sliding window')
#     p.add_argument('--predictor', type=str, choices=['lstm', 'gnn', 'transformer','transformer-gmm'], default='transformer-gmm',help='Predictor type')
#     p.add_argument('--setting', type=str, default='train',choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
#     p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
#     p.add_argument('--annotations_path', type=str, default="data/annotations", help='If annotations are placed in a location different from recomended')
#     p.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
#     p.add_argument('--normalize', default=True, type=bool, help='Normalize data, recommended True')
#     p.add_argument('--batch_size', type=int, default=64, help='Batch size')
#     p.add_argument('--device', type=str, default='cuda', help='device to run the model',choices=['cuda', 'cpu'])
#     p.add_argument('--seed', type=int, default=42, help='Seed for reproducibility -> set zero for random seed generation')

#     args = p.parse_args()
   

#     if args.setting == "evaluate" and not args.checkpoint:
#         print("No checkpoint provided")
#         exit
#     elif args.setting == "train" and not args.checkpoint:
#         args.checkpoint = f'GMM_transformer_P_{args.past_trajectory}_F_{args.future_trajectory}_W_{args.window_size}.pth'
#         # args.checkpoint = f'transformer_P_{args.past_trajectory}_F_{args.future_trajectory}_W_{args.window_size}.pth'
#     if args.device == "cuda" and not cuda.is_available():
#         args.device = "cpu"
#         print("Could not find GPU. Using CPU instead!")

    

#     # Print all arguments
#     for arg in vars(args):
#         if arg == "seed" and int(getattr(args, arg)) == 0:
#             print(f"{arg:20s}: seed not selected (random selection)")
#         else:
#             print(f"{arg:20s}: {getattr(args, arg)}")


#     # set seed for deterministic training -> # if seed is zero then don't set seed
#     if int(args.seed)>0: 
#         set_seeds(int(args.seed))

    
#     # Generate setting
#     ann_path = "../data/annotations" if not args.annotations_path else args.annotations_path
#     prd_ann_path = ann_path + "/prediction_annotations"
#     annotations = [prd_ann_path + '/' + f for f in os.listdir(prd_ann_path)]
#     splits = load_meta_data(ann_path + "/metadata.txt")
#     data_folder = generate_prediction_settings(args.past_trajectory, args.future_trajectory, splits, annotations, args.window_size)
    
#     # Create DataLoaders
#     # Transformer and GMM models use relative position rather than absolute position 
#     if args.predictor=='transformer' or args.predictor=='transformer-gmm': 
#         include_velocity = True
#     else:
#         include_velocity = False
        
#     tain_dataset = SeqDataset(data_folder,"train",include_velocity)
#     train_dataloader = DataLoader(tain_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
#     test_datatset = SeqDataset(data_folder, "test",include_velocity)
#     test_dataloader = DataLoader(test_datatset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)



#     # get mean and standard deviation
#     train_mean,train_std = tain_dataset.mean,tain_dataset.std
#     max_timestep_len  =  max(args.past_trajectory, args.future_trajectory)
   


#     # Train Predictor
#     if args.predictor=='transformer':
#         # Initialize model - > All parameters are the same as ModelConfig defaults except max_length
#         transformer = AttentionEMT(
#             max_length=max_timestep_len,
#             device=args.device
#         ).to(args.device)

#         transformer.train_model(args,train_dl=train_dataloader ,test_dl=test_dataloader,epochs=int(args.epochs),mean=train_mean,std=train_std)
    
#     if args.predictor=='transformer-gmm':
#         args.checkpoint = f'GMM_transformer_P_{args.past_trajectory}_F_{args.future_trajectory}_W_{args.window_size}.pth'
#         # Initialize model - > All parameters are the same as ModelConfig defaults except max_length
#         gmmTransformer = AttentionGMM(
#             past_trajectory=args.past_trajectory, future_trajectory=args.future_trajectory, device=args.device,normalize=args.normalize,
#         mean=train_mean,std=train_std).to(args.device)

#         gmmTransformer.train(train_dl=train_dataloader ,test_dl=test_dataloader,epochs=int(args.epochs))

        
 
           
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:08:42 2024

@author: nadya started it but murad messed it up
"""

import os
import cv2
import sys
import argparse
import numpy as np
import json    
from torch.utils.data import DataLoader
import torch.cuda as cuda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_meta_data, set_seeds
from utils import generate_prediction_settings
from utils import generate_intention_settings

from evaluation.distance_metrics import calculate_ade, calculate_fde
from dataloaders.seq_loader import SeqDataset
from dataloaders.frame_loader import GNNDataset
from models.rnn import RNNPredictor
from models.gnn import GCNPredictor, GATPredictor
from models.transformer import AttentionEMT
from models.tranformer_GMM import AttentionGMM
import torch
import torch.nn as nn

gnn_predictors = set(["gcn", "gat"])

def create_predictor(past_trajectory, future_trajectory, max_nodes, predictor, device, normalize, checkpoint_file):
    """
    Initializes and returns the appropriate predictor model based on the specified type.
    
    Args:
        past_trajectory (int): Number of past timesteps used for prediction.
        future_trajectory (int): Number of future timesteps to predict.
        max_nodes (int): Maximum number of nodes for GNN-based models.
        predictor (str): The type of predictor to initialize ('gcn', 'gat', 'transformer', etc.).
        device (str): Device on which the model should run ('cuda' or 'cpu').
        normalize (bool): Whether to normalize input features.
        checkpoint_file (str): Path to a checkpoint file if loading a pre-trained model.

    Returns:
        Model instance of the specified predictor type.
    """
    if predictor == "gcn":
        return GCNPredictor(past_trajectory, future_trajectory, max_nodes, device, normalize, checkpoint_file)
    elif predictor == "gat":
        return GATPredictor(past_trajectory, future_trajectory, max_nodes, device, normalize, checkpoint_file) 
    elif predictor == 'transformer':
        return AttentionEMT(past_trajectory, future_trajectory, device, normalize, checkpoint_file)
    elif predictor == 'transformer-gmm':
        return AttentionGMM(past_trajectory=past_trajectory, future_trajectory=future_trajectory, device=device, normalize=normalize, checkpoint_file=checkpoint_file)
    else:
        return RNNPredictor(past_trajectory, future_trajectory, device, normalize, checkpoint_file)
        

def create_dataset(data_folder, predictor, max_nodes, setting="train"):
    """
    Initializes and returns the appropriate dataset based on the predictor type.
    
    Args:
        data_folder (str): Path to the folder containing dataset files.
        predictor (str): The type of predictor to determine dataset type ('gcn', 'gat', etc.).
        max_nodes (int): Maximum number of nodes for GNN-based datasets.
        setting (str): Dataset mode ('train' or 'test').

    Returns:
        Dataset instance of the appropriate type.
    """
    if predictor in gnn_predictors:
        return GNNDataset(data_folder, max_nodes, setting=setting)
    else:
        return SeqDataset(data_folder, setting=setting)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('past_trajectory', type=int, help='Past Trajectory')
    p.add_argument('future_trajectory', type=int, help='Prediction Horizon')
    # p.add_argument('predictor', type=str, choices=['lstm', 'gcn', 'gat', 'transformer', 'transformer_gmm'], help='Predictor type')
    p.add_argument('--window_size', default=1, type=int, help='Sliding window')
    p.add_argument('--max_nodes', type=int, default=40, help='Maximum number of nodes for GNN model')
    p.add_argument('--predictor', type=str, default='transformer-gmm', choices=['lstm', 'gcn', 'gat', 'transformer','transformer-gmm'], help='Predictor type')
    p.add_argument('--setting', type=str, default='train',choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
    p.add_argument('--annotations_path', type=str, help='If annotations are placed in a location different from recommended')
    p.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
    p.add_argument('--normalize', default=True, type=bool, help='Normalize data, recommended True')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size')
    p.add_argument('--device', type=str, default='cuda', help='Device to run the model',choices=['cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42, help='Seed for reproducibility -> set zero for random seed generation')

    args = p.parse_args()
    
    set_seeds(int(args.seed))
    
    # Generate setting
    ann_path = "../data/annotations" if not args.annotations_path else args.annotations_path
    prd_ann_path = ann_path + "/prediction_annotations"
    annotations = [prd_ann_path + '/' + f for f in os.listdir(prd_ann_path)]
    splits = load_meta_data(ann_path + "/metadata.txt")
    generating_setting = "frame" if args.predictor in gnn_predictors else "seq"
    data_folder = generate_prediction_settings(args.past_trajectory, 
                                               args.future_trajectory, 
                                               splits, annotations, 
                                               args.window_size, 
                                               generating_setting)
    
    # Create Dataset
    train_dataset = create_dataset(data_folder, args.predictor, args.max_nodes, setting="train")
    test_dataset = create_dataset(data_folder, args.predictor, args.max_nodes, setting="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    #Train and Evaluate Predictor 
    predictor = create_predictor(args.past_trajectory, 
                                args.future_trajectory, 
                                args.max_nodes, 
                                args.predictor, 
                                args.device,
                                args.normalize,
                                args.checkpoint)
    predictor.train(train_loader,test_loader) 
    predictor.evaluate(test_loader)