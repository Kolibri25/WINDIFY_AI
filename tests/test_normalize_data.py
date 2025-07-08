import xarray as xr
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.normalize_data import normalize_data


def test_normalization(input_path, var_name, atol=1e-5):
    ds = xr.open_dataset(input_path)
    ds_norm, mean, std = normalize_data(ds, var_name)

    data = ds_norm[var_name].values
    new_mean = data.mean()
    new_std = data.std()

    mean_ok = np.isclose(new_mean, 0.0, atol=atol)
    std_ok = np.isclose(new_std, 1.0, atol=atol)

    print(f"{var_name} | mean: {new_mean:.5f} | std: {new_std:.5f}")
    assert mean_ok, f"Mean not ~0 (got {new_mean})"
    assert std_ok, f"Std not ~1 (got {new_std})"

    print("✅ Normalization test passed.")

if __name__ == "__main__":
    test_normalization("data/barra_downloads/uas_gesamt.nc", "uas")
    test_normalization("data/barra_downloads/vas_gesamt.nc", "vas")
