import xarray as xr
import glob
import os

def collect_files(directory):
    files = glob.glob(os.path.join(directory, "ERA5_*_merged.nc"))
    files = sorted(files)
    return files

def concat_era5_files(parent_directory):
    files = collect_files(parent_directory)
    if not files:
        print("No ERA5 files found.")
        return

    save_path = os.path.join(parent_directory, "data/ERA5/combined_era5.nc")
    print(f"Saving to: {save_path}")

    if os.path.exists(save_path):
        print(f"{save_path} already exists. Aborting to avoid overwrite.")
        return

    datasets = []
    for file in files:
        try:
            ds = xr.open_dataset(file)
            datasets.append(ds)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if datasets:
        combined = xr.concat(datasets, dim='valid_time')
        combined = combined.sortby('valid_time')
        combined.to_netcdf(save_path)
        print(f"Combined dataset saved as {save_path}")
    else:
        print("No datasets to concatenate.")

if __name__ == "__main__":
    # Adjust to your actual project structure
    PARENT_DIRECTORY = "/Users/ayeshakhan/WINDIFY_AI/"
    concat_era5_files(PARENT_DIRECTORY)
