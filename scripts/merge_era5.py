import xarray as xr
import os
import glob
from collections import defaultdict



PARENT_DIRECTORY = "/Users/ayeshakhan/WINDIFY_AI/data/ERA5/"
SAVE_PATH = os.path.join(PARENT_DIRECTORY, "combined_era5.nc")


# Helper: returns True if datetime is in Jan, Oct, Nov, Dec
def is_valid_month(dt_array):
    months = dt_array.dt.month
    return (months == 1) | (months == 10) | (months == 11) | (months == 12)

# Step 1: Collect all .nc files from all subfolders
all_files = glob.glob(os.path.join(PARENT_DIRECTORY, "ERA5_*_daily", "*.nc"))
print(f"Found {len(all_files)} .nc files.")

# Step 2: Organize datasets by year
year_to_datasets = defaultdict(list)

for file in sorted(all_files):
    try:
        ds = xr.open_dataset(file)
        ds = ds.sel(valid_time=is_valid_month(ds.valid_time))
        year = int(str(ds.valid_time[0].dt.year.values))
        year_to_datasets[year].append((file, ds))
        print(f"Loaded {file} → {year}, shape: {ds.sizes['valid_time']}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Step 3: Merge per year and concatenate
all_years = []
for year in range(1979, 2025):
    if year not in year_to_datasets:
        print(f"⚠️  No data found for year {year}")
        continue

    entries = year_to_datasets[year]
    if len(entries) < 5:
        print(f"⚠️  Year {year} is missing variables: only {len(entries)} files:")
        for f, _ in entries:
            print(f"    - {os.path.basename(f)}")
        continue

    try:
        datasets = [ds for _, ds in entries]
        combined = xr.merge(datasets)

        # Sort and validate
        combined = combined.sortby("valid_time")
        if len(combined.valid_time) != 123:
            raise ValueError(f"Year {year} has {len(combined.valid_time)} time steps (expected 123)")

        all_years.append(combined)
        print(f"✅ Year {year} merged successfully.")

    except Exception as e:
        print(f"❌ Failed to merge year {year}: {e}")

# Step 4: Final concat and save
if all_years:
    final = xr.concat(all_years, dim="valid_time")
    final = final.sortby("valid_time")
    final.to_netcdf(SAVE_PATH)
    print(f"\n🎉 Combined dataset saved to {SAVE_PATH} with {len(final.valid_time)} time steps.")
else:
    print("❌ No valid yearly datasets to combine.")

