import pandas as pd
import xarray as xr

# Load actual time values
ds = xr.open_dataset("your_era5_subset.nc")
ac_tim = pd.to_datetime(ds['time'].values)
ac_tim = pd.Series(ac_tim).drop_duplicates().sort_values().reset_index(drop=True)

# Define expected months (e.g., Jan, Apr, Jul, Oct)
expected_months = [1, 10, 11, 12]

# Create filtered expected date range
full_range = pd.date_range(start=ac_tim.min(), end=ac_tim.max(), freq='D')
expected_times = full_range[full_range.month.isin(expected_months)]

# Compare
missing = expected_times.difference(ac_tim)

print(f"Expected dates (filtered): {len(expected_times)}")
print(f"Actual dates:              {len(ac_tim)}")
print(f"Missing dates:             {len(missing)}")
print(missing)
