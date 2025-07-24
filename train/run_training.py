
import os
import sys
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.models.unet_model import UNet
from src.data.dataset import WindDownscalingDataset
from src.train.trainer import train_model

import torch
import platform

if torch.backends.mps.is_available() and platform.system() == "Darwin":
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ------------------
# 1. Load training and validation Data
# ------------------

# Paths to input and target tensors
input_path = "data/processed_data/input_train.pt"
target_path = "data/processed_data/target_train.pt"

input_val = "data/processed_data/input_val.pt"
target_val = "data/processed_data/target_val.pt"

val_dataset = WindDownscalingDataset(
    input_tensor=torch.load(input_val),
    target_tensor=torch.load(target_val)
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Load tensors
input_tensor = torch.load(input_path)
target_tensor = torch.load(target_path)
# Check if tensors are loaded correctly
if input_tensor is None or target_tensor is None:
    raise ValueError("Failed to load input or target tensors.")
print("✅ Loaded input + target tensors:", input_tensor.shape, target_tensor.shape)
print("NaNs in target:", torch.isnan(target_tensor).sum().item(), "out of", target_tensor.numel())

print("📊 Input stats — min:", input_tensor.min().item(), "max:", input_tensor.max().item())
print("📊 Target stats — min:", target_tensor.min().item(), "max:", target_tensor.max().item())

assert input_tensor.shape[0] == target_tensor.shape[0], "Input and target tensors must have the same number of samples."

# ------------------
# 2. Create Dataset and DataLoader
# ------------------
dataset = WindDownscalingDataset(input_tensor, target_tensor)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ------------------
# 3. Define Model
# ------------------
model = UNet(n_channels=2, n_classes=2, 
             bilinear=True)

# ------------------
# 4. Define Training Parameters
# ------------------
optimizer_fn = torch.optim.Adam
optimizer_kwargs = {'lr': 1e-4}
loss_fn = nn.MSELoss()
scheduler_fn = torch.optim.lr_scheduler.StepLR
scheduler_kwargs = {'step_size': 10, 'gamma': 0.1}
# Save directory for checkpoints
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
# ------------------
# 5. Train the Model
# ------------------
print("✅ Starting training...")

model, losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    optimizer_fn=optimizer_fn,
    optimizer_kwargs=optimizer_kwargs,
    loss_fn=loss_fn,
    scheduler_fn=scheduler_fn,
    scheduler_kwargs=scheduler_kwargs,
    device=device,
    save_dir=SAVE_DIR
)
