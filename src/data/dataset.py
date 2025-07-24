import torch
from torch.utils.data import Dataset

class WindDownscalingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor, transform=None):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
            target_tensor (torch.Tensor): Target tensor of shape (N, C, H, W).
            transform (callable, optional): Optional transform to be applied on the input tensor.
        """
        if input_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError("Input and target tensors must have the same number of samples.")
        assert input_tensor.shape[2] == target_tensor.shape[2], "Input and target tensors must have the same height."
        assert input_tensor.shape[3] == target_tensor.shape[3], "Input and target tensors must have the same width."
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __len__(self):
        return self.input_tensor.shape[0]
    
    def __getitem__(self, idx):
        x = self.input_tensor[idx]
        y = self.target_tensor[idx]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
    