import numpy as np
import xarray as xr
import os

def load_and_save_barra_c2():
    data_folder_ua = r"data\BARRA_C2\BARRA_C2_ua10"
    data_folder_va = r"data\BARRA_C2\BARRA_C2_va10"

    months = ["10", "11", "12", "01"]
    years = [str(year) for year in range(1979, 2025)]

    ua_datasets = []
    va_datasets = []

    for year in years:
        for month in months:
            yyyymm = year + month
            filename_ua = f"ua10_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{yyyymm}-{yyyymm}.nc"
            filepath_ua = os.path.join(data_folder_ua, filename_ua)
            filename_va = f"va10_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{yyyymm}-{yyyymm}.nc"
            filepath_va = os.path.join(data_folder_va, filename_va)

            if os.path.exists(filepath_ua):
                ds_ua = xr.open_dataset(filepath_ua, decode_cf=False)
                ds_ua = ds_ua.isel(lon=slice(0, -7), lat=slice(5, -6))
                ua_datasets.append(ds_ua)
                print(f"{filename_ua} loaded successfully.")
            else:
                print(f"Datafile not found: {filename_ua}")

            if os.path.exists(filepath_va):
                ds_va = xr.open_dataset(filepath_va, decode_cf=False)
                ds_va = ds_va.isel(lon=slice(0, -7), lat=slice(5, -6))
                va_datasets.append(ds_va)
                print(f"{filename_va} loaded successfully.")
            else:
                print(f"Datafile not found: {filename_va}")

    ua_combined = xr.concat(ua_datasets, dim="time")
    va_combined = xr.concat(va_datasets, dim="time")

    ua_combined.to_netcdf("ua10_gesamt.nc")
    va_combined.to_netcdf("va10_gesamt.nc")

    print("BARRA_C2 combined datasets saved as ua10_gesamt.nc and va10_gesamt.nc.")


def load_and_save_era5():
    base_folder = r"data\ERA5"
    years = [str(year) for year in range(1979, 2025)]

    daily_filenames = [
        "2m_temperature_stream-oper_daily-mean.nc",
        "mean_sea_level_pressure_0_daily-mean.nc",
        "mean_wave_direction_1_daily-mean.nc",
        "10m_u_component_of_wind_stream-oper_daily-mean.nc",
        "10m_v_component_of_wind_0_daily-mean.nc"
    ]

    for year in years:
        datasets = []

        for fname in daily_filenames:
            file_path = os.path.join(base_folder, f"ERA5_{year}_daily", fname)

            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path, decode_cf=False)
                # slice dimensions if needed
                # ds = ds.isel(lon=slice(0, -7), lat=slice(5, -6))
                datasets.append(ds)
                print(f"{file_path} loaded successfully.")
            else:
                print(f"Datafile not found: {file_path}")

        if datasets:
            merged_ds = xr.merge(datasets)
            save_path = f"ERA5_{year}_merged.nc"
            merged_ds.to_netcdf(save_path)
            print(f"Merged ERA5 dataset saved as {save_path}.")


if __name__ == "__main__":
    print("Start loading and saving BARRA_C2 data ...")
    load_and_save_barra_c2()
    print("Start loading and saving ERA5 data ...")
    load_and_save_era5()
    print("All data processed and saved.")

