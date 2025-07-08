def crop_to_multiple_of_16(ds, lat_name='lat', lon_name='lon'):
    """
    Crop dataset so that lat/lon dimensions are divisible by 16.
    Selects a centered subregion that satisfies the constraint.

    Args:
        ds (xarray.Dataset): The dataset to crop
        lat_name (str): Name of the latitude dimension
        lon_name (str): Name of the longitude dimension

    Returns:
        xarray.Dataset: Cropped dataset
    """
    lat_dim = ds[lat_name].size
    lon_dim = ds[lon_name].size

    lat_crop = lat_dim % 16
    lon_crop = lon_dim % 16

    if lat_crop == 0 and lon_crop == 0:
        return ds  # already divisible

    lat_start = lat_crop // 2
    lon_start = lon_crop // 2

    return ds.isel(
        **{
            lat_name: slice(lat_start, lat_start + lat_dim - lat_crop),
            lon_name: slice(lon_start, lon_start + lon_dim - lon_crop)
        }
    )
