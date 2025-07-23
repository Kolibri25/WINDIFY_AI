# scripts/preprocess_data.py
"""
Crops, interpolates, and normalizes ERA5 and BARRA data.
Outputs: processed input and target PyTorch tensors.
"""

import sys
import os

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

print(scripts_path)

import download_barra as dnb
import load_data as ld
from model_input import create_uv_tensor, create_uv_tensor_era5
from crop_to_div16 import crop_to_multiple_of_16
from normalize_data import normalize_data


import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt

uas_ds = xr.open_dataset('data/BARRA-C2/uas_gesamt.nc')
vas_ds = xr.open_dataset('data/BARRA-C2/vas_gesamt.nc')

# load input data (ERA5)

era5_ds = xr.open_dataset('data/ERA5/combined_era5.nc')
# drop duplicates
era5_ds = era5_ds.drop_duplicates(dim='valid_time')

print("NaNs in uas_ds:", np.isnan(uas_ds["uas"].values).sum())
print("NaNs in vas_ds:", np.isnan(vas_ds["vas"].values).sum())

# shape check
print("uas_ds shape:", uas_ds.sizes)
print("vas_ds shape:", vas_ds.sizes)

vas_ds = crop_to_multiple_of_16(vas_ds, lat_name='lat', lon_name='lon')

# interpolation due to misalignment
uas_ds = uas_ds.interp_like(vas_ds, method='nearest')  # match shape to cropped vas
print("NaNs in interpolated uas:", np.isnan(uas_ds["uas"].values).sum())
uas_ds = crop_to_multiple_of_16(uas_ds, lat_name='lat', lon_name='lon')  # now crop vas


uas_ds, uas_mean, uas_std = normalize_data(uas_ds,"uas")
vas_ds, vas_mean, vas_std = normalize_data(vas_ds,"vas")

# Interpolate ERA5 data to match BARRA-C2 grid


era5_interp = era5_ds.interp(
    latitude=uas_ds.lat,
    longitude=uas_ds.lon,
    method='nearest'
)

# normalise input data

u10_ds, u10_mean, u10_std = normalize_data(era5_interp,"u10")
v10_ds, v10_mean, v10_std = normalize_data(era5_interp,"v10")

# convert to PyTorch tensors

print("Shape of uas_ds:", uas_ds.lon.shape)
print("Shape of vas_ds:", vas_ds.lon.shape)

target = create_uv_tensor(uas_ds,vas_ds)
print("NaNs in target:", torch.isnan(target).sum())

input = create_uv_tensor_era5(u10_ds["u10"],v10_ds["v10"])

# Check shapes
print(target.shape)
print(input.shape)

output_path = "data/processed_data/"

# Ensure trailing slash and directory exists
os.makedirs(output_path, exist_ok=True)

# Use os.path.join to avoid hardcoding slashes
torch.save(target, os.path.join(output_path, 'target.pt'))
torch.save(input, os.path.join(output_path, 'input.pt'))