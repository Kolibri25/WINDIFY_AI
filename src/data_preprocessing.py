import scripts.download_barra as dnb
import scripts.load_data as ld
from scripts.model_input import create_uv_tensor, create_uv_tensor_era5
from scripts.crop_to_div16 import crop_to_multiple_of_16
from scripts.normalize_data import normalize_data

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt

uas_ds = xr.open_dataset('data/BARRA-C2/uas_gesamt.nc')
vas_ds = xr.open_dataset('data/BARRA-C2/vas_gesamt.nc')

# load input data (ERA5)

era5_ds = xr.open_dataset('data/ERA5/combined_era5.nc')


uas_ds = crop_to_multiple_of_16(uas_ds, lat_name='lat', lon_name='lon')
vas_ds = crop_to_multiple_of_16(vas_ds, lat_name='lat', lon_name='lon')

uas_ds, uas_mean, uas_std = normalize_data(uas_ds,"uas")
vas_ds, vas_mean, vas_std = normalize_data(vas_ds,"vas")

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
print("uas_ds longitude values:", uas_ds.lon.values)
print("vas_ds longitude values:", vas_ds.lon.values)

# Interpolate vas_ds to match uas_ds longitude coordinates
vas_ds_interp = vas_ds.interp(lon=uas_ds.lon, method='nearest')

target = create_uv_tensor(uas_ds,vas_ds_interp)
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