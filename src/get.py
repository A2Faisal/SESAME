import os
import xarray as xr
import pandas as pd


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

def get_regional_data(df, regions_df, region_name='Region 1'):
    
    # Merge the dataframes based on ISO3 code
    merged_df = pd.merge(df, regions_df, left_on='ISO3', right_on='ISO-alpha3 code', how='left')

    # Check for any ISO3 codes that are not matching
    unmatched = merged_df[merged_df[region_name].isna()]
    print("Unmatched ISO3 codes:\n", unmatched['ISO3'].unique())

    # Group by the desired region and sum the columns
    regional_sum = merged_df.groupby('Region 1').sum(numeric_only=True)
    regional_sum = regional_sum.drop(columns=["M49 code"])

    return regional_sum


def get_netcdf_info(netcdf_path, variable_name=None):

    ds = xr.open_dataset(netcdf_path)

    # Get the list of variables using list comprehension
    var_short_name = [var for var in ds.data_vars if variable_name is None or var.startswith(variable_name)]

    # Extract dimensions
    var_dims = list(ds.dims)

    # Extract long names and units using list comprehensions
    var_long_name = [ds[var].attrs.get('long_name', None) for var in var_short_name]
    var_unit = [ds[var].attrs.get('units', ds[var].attrs.get('unit', None)) for var in var_short_name]

    # Check if 'time' variable exists in the dataset, if True then extract time values as strings
    if 'time' in ds:
        var_time = ds["time"].values.astype(str).tolist()
    else:
        var_time = None

    return var_dims, var_short_name, var_long_name, var_unit, var_time
