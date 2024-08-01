import os
import xarray as xr


def identify_lat_lon_names(netcdf_path):
    """
    Uses common names for latitude and longitude coordinates to identify
    the corresponding dimensions in a NetCDF dataset. It is designed to work with datasets
    represented using the xarray library.

    Parameters
    ----------
    netcdf_path : str
        The file path to the NetCDF dataset.

    Returns
    -------
    tuple
        A tuple containing the identified x (longitude) and y (latitude) dimensions.
    """
    ds = xr.open_dataset(netcdf_path)

    common_lat_names = [
        'lat',
        'latitude',
        'y',
        'south_north',
        'grid_latitude',
        'latitudes',
        'Y',
        'Y_AXIS'
    ]

    common_lon_names = [
        'lon',
        'longitude',
        'x',
        'west_east',
        'grid_longitude',
        'longitudes',
        'X',
        'X_AXIS'
    ]

    # Identify latitude and longitude coordinates
    lat_coord = next((coord for coord in ds.coords if coord in common_lat_names), None)
    lon_coord = next((coord for coord in ds.coords if coord in common_lon_names), None)

    # Determine x and y dimensions based on latitude and longitude coordinates
    x_dimension = lon_coord
    y_dimension = lat_coord

    return x_dimension, y_dimension