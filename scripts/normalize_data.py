import xarray as xr
import torch

def normalize_data(ds, var_name):
    """
    Normalize the specified variable in the dataset using z-score normalization.
    This function computes the mean and standard deviation of the variable
    and normalizes it to have a mean of 0 and standard deviation of 1.
    If the standard deviation is zero, it raises an error since normalization is not possible.
    This is useful for preparing data for machine learning models or statistical analysis.

    Args:
        ds (xarray.Dataset): The dataset containing the variable to normalize.
        var_name (str): The name of the variable to normalize.

    Returns:
        xarray.Dataset: The dataset with the normalized variable.
    """
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    # Extract the variable data
    data = ds[var_name]

    # Calculate mean and standard deviation
    mean = data.mean()
    std = data.std()
    if std == 0 or torch.isnan(torch.tensor(std.values)):
        raise ValueError(f"Standard deviation of '{var_name}' is zero or NaN, normalization not possible.")
    # Normalize the data
    normalized_data = (data - mean) / std
    # Update the dataset with the normalized variable
    ds[var_name] = normalized_data
    ds[var_name].attrs['units'] = 'normalized'  # Update units to indicate normalization
    return ds, mean, std

def normalize_dataset(input_path, output_path, var_name):
    """
    Normalize a variable in an xarray dataset and save the result.
    Args:
        input_path (str): Path to the input dataset file.
        output_path (str): Path to save the normalized dataset.
        var_name (str): Name of the variable to normalize.
    """

    # Load the dataset
    ds = xr.open_dataset(input_path)

    # Normalize the specified variable
    ds_normalized = normalize_data(ds, var_name)

    # Save the normalized dataset
    ds_normalized.to_netcdf(output_path)
    print(f"Normalized dataset saved to {output_path}")

if __name__ == "__main__":
    
    input_path = "path/to/your/input_dataset.nc"  # Replace with your input dataset path
    output_path = "path/to/your/output_normalized_dataset.nc"    # Replace with your desired output path
    var_name = "variable_to_normalize"  # Replace with the variable you want to normalize

    normalize_dataset(input_path, output_path, var_name)