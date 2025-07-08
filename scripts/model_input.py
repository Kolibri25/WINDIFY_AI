import torch
import numpy as np
import xarray as xr


##############################################
# BARRA_C2 Tensor with ua and va components  #
##############################################

def create_uv_tensor(ua_ds, va_ds, ua_varname='uas', va_varname='vas'):
    """
    Combines ua and va datasets into a PyTorch tensor with shape:
    (time, 2, height=lat, width=lon)

    Args:
        ua_ds (xarray.Dataset): Dataset with ua variable.
        va_ds (xarray.Dataset): Dataset with va variable.
        ua_varname (str): Name of the ua variable in the dataset.
        va_varname (str): Name of the va variable in the dataset.

    Returns:
        torch.Tensor: Tensor of shape (time, 2, height, width)
    """
    assert ua_ds.time.equals(va_ds.time), "Time dimensions do not match"
    assert ua_ds.lat.equals(va_ds.lat), "Latitude dimensions do not match"
    assert ua_ds.lon.equals(va_ds.lon), "Longitude dimensions do not match"

    ua_array = ua_ds[ua_varname].values  # shape: (time, lat, lon)
    va_array = va_ds[va_varname].values

    # Stack along channel axis (ua, va) → shape: (time, 2, lat, lon)
    combined = np.stack([ua_array, va_array], axis=1)
    return torch.from_numpy(combined).float()




#################################################
# ERA5 Tensor with u and v components           #
#################################################

def create_uv_tensor_era5(u_array, v_array):
    """
    Creates a PyTorch tensor from u and v wind arrays (e.g. ERA5).

    Args:
        u_array (np.ndarray or xarray.DataArray): u-component (time, lat, lon)
        v_array (np.ndarray or xarray.DataArray): v-component (time, lat, lon)

    Returns:
        torch.Tensor: Tensor of shape (time, 2, height, width)
    """
    # Ensure they’re both NumPy arrays
    if hasattr(u_array, 'values'):
        u_array = u_array.values
    if hasattr(v_array, 'values'):
        v_array = v_array.values

    assert u_array.shape == v_array.shape, "Shape mismatch between u and v arrays"

    combined = np.stack([u_array, v_array], axis=1)  # (time, 2, lat, lon)
    return torch.from_numpy(combined).float()


#########################################
# Split tensor into train/val/test sets #
#########################################

def split_tensor(tensor, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Splits a 4D tensor (time, channels, height, width) into
    training, validation, and test sets.

    Based on Addision et al. (2025):
        - 70% Training: model development
        - 15% Validation: model selection
        - 15% Test: final evaluation

    Args:
        tensor (torch.Tensor): Input tensor of shape (time, channels, height, width)
        train_frac (float): Fraction for training set (default: 0.7)
        val_frac (float): Fraction for validation set (default: 0.15)
        test_frac (float): Fraction for test set (default: 0.15)

    Returns:
        Tuple[train, val, test]: Tensors split accordingly
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5, "Fractions must sum to 1"

    n = tensor.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = tensor[:n_train]
    val = tensor[n_train:n_train + n_val]
    # take rest of the tensor as test data
    test = tensor[n_train + n_val:]          

    return train, val, test



###############################
# Example usage and saving    #
###############################

def main():
    # Load BARRA_C2 data
    ua_ds = xr.open_dataset("ua10_gesamt.nc", decode_cf=False)
    va_ds = xr.open_dataset("va10_gesamt.nc", decode_cf=False)

    # Create tensor
    tensor = create_uv_tensor(ua_ds, va_ds, ua_varname='ua10', va_varname='va10')
    print("BARRA_C2 Tensor shape:", tensor.shape)

    # Split into train/val/test sets
    train, val, test = split_tensor(tensor)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    # Optional: Save as .pt files
    #torch.save(train, "barra_train.pt")
    #torch.save(val, "barra_val.pt")
    #torch.save(test, "barra_test.pt")
    #print("Tensors saved as barra_train.pt, barra_val.pt, barra_test.pt")



    # Example for ERA5 (uncomment if needed)
    # merged_ds = xr.open_dataset("ERA5_1980_merged.nc", decode_cf=False)
    # tensor_era5 = create_uv_tensor_era5(
    #     merged_ds,
    #     u_varname='10m_u_component_of_wind_stream_oper',
    #     v_varname='10m_v_component_of_wind_0'
    # )
    # print("ERA5 Tensor shape:", tensor_era5.shape)