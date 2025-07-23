import sys
import os

# Add /Users/ayeshakhan/WINDIFY_AI/src to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from data.dataset import WindDownscalingDataset

# Load split tensors
input_train = torch.load("data/processed_data/input_train.pt")
target_train = torch.load("data/processed_data/target_train.pt")

# Wrap in dataset
train_dataset = WindDownscalingDataset(input_train, target_train)

# Wrap in dataloader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Try iterating once
for xb, yb in train_loader:
    print("X:", xb.shape)
    print("Y:", yb.shape)
    break
