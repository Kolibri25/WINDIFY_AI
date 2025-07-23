import torch
import os
from model_input import split_tensor

# paths to processed input/target tensors
INPUT_PATH = "data/processed_data/input.pt"
TARGET_PATH = "data/processed_data/target.pt"

SAVE_DIR = "data/processed_data/"
os.makedirs(SAVE_DIR, exist_ok=True)

# load tensors
input_tensor = torch.load(INPUT_PATH)
target_tensor = torch.load(TARGET_PATH)

# Check if tensors are loaded correctly
if input_tensor is None or target_tensor is None:
    raise ValueError("Failed to load input or target tensors.")

assert input_tensor.shape[0] == target_tensor.shape[0], "Input and target tensors must have the same number of samples."
assert input_tensor.shape[2] == target_tensor.shape[2], "Input and target tensors must have the same height."
assert input_tensor.shape[3] == target_tensor.shape[3], "Input and target tensors must have the same width."

# Split the tensors into training, validation, and test sets
input_train, input_val, input_test = split_tensor(input_tensor)
target_train, target_val, target_test = split_tensor(target_tensor)

# Save the split tensors
torch.save(input_train, os.path.join(SAVE_DIR, "input_train.pt"))
torch.save(input_val, os.path.join(SAVE_DIR, "input_val.pt"))
torch.save(input_test, os.path.join(SAVE_DIR, "input_test.pt"))
torch.save(target_train, os.path.join(SAVE_DIR, "target_train.pt"))
torch.save(target_val, os.path.join(SAVE_DIR, "target_val.pt"))
torch.save(target_test, os.path.join(SAVE_DIR, "target_test.pt"))
print("Data split and saved successfully.")

