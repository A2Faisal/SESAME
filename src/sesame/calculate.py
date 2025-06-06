import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import pyproj
import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def calculate_geometry_attributes(input_gdf, column_name=None):
    """
    Calculate area or length for each geometry in the GeoDataFrame and store it in a specified column.
    For polygons, calculates the area in square meters using NSIDC EASE-Grid 2.0 (EPSG:6933).
    For lines, calculates the length in meters using NSIDC EASE-Grid 2.0 (EPSG:6933).

    Parameters
    ----------
    input_gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing geometries.
    column_name : str, optional
        Column name to store the calculated values (either area or length). 
        If None, 'area_m2' or 'length_m' will be used based on the geometry type.

    Returns
    -------
    gpd.GeoDataFrame
        The updated GeoDataFrame with the specified column.
    """
    # Copy the GeoDataFrame to avoid modifying the original
    gdf = input_gdf.copy()

    # Determine the appropriate column name if not provided
    if column_name is None:
        # Default to 'area_m2' or 'length_m' based on geometry type
        if any(isinstance(geom, (Polygon, MultiPolygon)) for geom in gdf.geometry):
            column_name = 'area_m2'
        elif any(isinstance(geom, (LineString, MultiLineString)) for geom in gdf.geometry):
            column_name = 'length_m'
        else:
            column_name = 'geometry_value'  # Default to a generic name if no polygons or lines are found

    # Project to NSIDC EASE-Grid 2.0 for area calculation in square meters
    projected_gdf = gdf.to_crs(epsg=6933)

    # Initialize list to store the calculated values
    values = []

    # Calculate area or length for each geometry
    for i, geom in enumerate(gdf.geometry):
        if geom.is_valid:
            if isinstance(geom, (Polygon, MultiPolygon)):
                projected_geom = projected_gdf.geometry[i]
                area = projected_geom.area
                values.append(np.float64(area))
            elif isinstance(geom, (LineString, MultiLineString)):
                projected_geom = projected_gdf.geometry[i]
                length = projected_geom.length
                values.append(np.float64(length))
            else:
                values.append(None)
        else:
            try:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    # Attempt to fix invalid polygon geometry with buffer(0)
                    fixed_geom = geom.buffer(0)
                    if fixed_geom.is_valid:
                        projected_geom = gpd.GeoSeries([fixed_geom], crs=gdf.crs).to_crs(epsg=6933).geometry[0]
                        area = projected_geom.area
                        values.append(np.float64(area))
                    else:
                        raise ValueError("Fixed geometry is still invalid")
                else:
                    # For invalid line geometries, just append None (or handle differently if needed)
                    values.append(None)
            except Exception as e:
                # Log the error and append None values
                print(f"Invalid geometry could not be fixed: {geom}")
                print(f"Error: {e}")
                values.append(None)

    # Add the calculated attributes to the GeoDataFrame
    gdf[column_name] = values

    # Drop the column if it contains only None values
    if gdf[column_name].isnull().all():
        gdf.drop(columns=[column_name], inplace=True)

    return gdf


def calculate_geodetic_pixel_area(lon, lat, pixel_width_deg, pixel_height_deg):
    geod = pyproj.Geod(ellps="WGS84")
    lons = [lon - pixel_width_deg / 2, lon + pixel_width_deg / 2, lon + pixel_width_deg / 2, lon - pixel_width_deg / 2, lon - pixel_width_deg / 2]
    lats = [lat - pixel_height_deg / 2, lat - pixel_height_deg / 2, lat + pixel_height_deg / 2, lat + pixel_height_deg / 2, lat - pixel_height_deg / 2]
    area, _ = geod.polygon_area_perimeter(lons, lats)[:2]
    return abs(area) #/ 1e6  Convert from square meters to square kilometers


def calculate_grid_resolution(resolution):
    """
    Calculate the number of latitude and longitude grid cells based on the given resolution.

    Parameters
    ----------
    resolution : int, float, or str
        The resolution can be provided as a numeric value (int or float), or as a string
        in the format '<value> degree(s)', where <value> is a numeric value.

    Returns
    -------
    tuple
        A tuple containing the calculated number of latitude and longitude grid cells.

    Raises
    ------
    ValueError
        If the resolution is not in a valid format or cannot be converted to a numeric value.
    """

    try:
        # Convert the resolution to a numeric value (degrees)
        if isinstance(resolution, (int, float)):
            degrees = float(resolution)
        else:
            # If resolution is a string, extract the numeric value
            degrees = float(resolution.split()[0])

        # Calculate the number of latitude and longitude grid cells
        num_lat, num_lon = int(180 / degrees), int(360 / degrees)

        return num_lat, num_lon
    except (ValueError, AttributeError):
        # Raise an error if the resolution is not in a valid format
        raise ValueError("Resolution should be in the format '<value> degree(s)' or a numeric value")
        

def sum_variables(dataset, variables=None, new_variable_name=None, time=None):
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif isinstance(dataset, xr.Dataset):
        dataset = dataset
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")   
    
    # Ensure all specified variables are in the dataset
    for var in variables:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    if variables is None:
        exclude_vars = ("grid_area", "land_frac")
        variables = [var for var in dataset.data_vars if not var.startswith(exclude_vars)]
    
    # Ensure all specified variables are in the dataset
    for var in variables:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    # Fill NaNs with zero before summing
    filled_vars = [dataset[var].fillna(0) for var in variables]
    
    # Sum the specified variables
    summed_data = sum(filled_vars)
    
    # Convert resulting zeros back to NaNs
    summed_data = summed_data.where(summed_data != 0, other=np.nan)
    
    if new_variable_name:
        # Create a new dataset with the summed variable
        summed_dataset = xr.Dataset({new_variable_name: summed_data})
    else:
        summed_dataset = xr.Dataset({'summed_variable': summed_data})
    
    if time is not None:
        time_coord = pd.to_datetime(time)
        summed_dataset = summed_dataset.expand_dims(time=[time_coord])
    
    return summed_dataset


def subtract_variables(dataset,variable1, variable2, new_variable_name=None, time=None):
    # Load dataset (accept path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif not isinstance(dataset, xr.Dataset):
        raise TypeError("`dataset` must be an xarray.Dataset or a path to a NetCDF file.")

    # Select time if specified
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')

    # Ensure both variables are in the dataset
    for var in [variable1, variable2]:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    # Fill NaNs with zero before subtracting
    filled_variable1 = dataset[variable1].fillna(0)
    filled_variable2 = dataset[variable2].fillna(0)
    
    # Subtract variable2 from variable1
    result_data = filled_variable1 - filled_variable2
    
    # Convert resulting zeros back to NaNs
    result_data = result_data.where(result_data != 0, other=np.nan)
    
    if new_variable_name:
        # Create a new dataset with the resulting variable
        result_dataset = xr.Dataset({new_variable_name: result_data})
    else:
        result_dataset = xr.Dataset({'result_variable': result_data})
    
    if time is not None:
        time_coord = pd.to_datetime(time)
        result_dataset = result_dataset.expand_dims(time=[time_coord])
    
    return result_dataset


def divide_variables(dataset, variable1, variable2, new_variable_name=None, time=None):
    
    # Load dataset (accept path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif not isinstance(dataset, xr.Dataset):
        raise TypeError("`dataset` must be an xarray.Dataset or a path to a NetCDF file.")

    # Select time if specified
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')

    # Ensure both variables are in the dataset
    for var in [variable1, variable2]:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    # Fill NaNs with zero before dividing
    filled_variable1 = dataset[variable1].fillna(0)
    filled_variable2 = dataset[variable2].fillna(0)
    
    # Divide variable1 by variable2
    with np.errstate(divide='ignore', invalid='ignore'):
        result_data = xr.where(filled_variable2 != 0, filled_variable1 / filled_variable2, np.nan)
    
    # Convert resulting zeros back to NaNs
    result_data = result_data.where(result_data != 0, other=np.nan)
    
    if new_variable_name:
        # Create a new dataset with the resulting variable
        result_dataset = xr.Dataset({new_variable_name: result_data})
    else:
        result_dataset = xr.Dataset({'result_variable': result_data})
    
    if time is not None:
        time_coord = pd.to_datetime(time)
        result_dataset = result_dataset.expand_dims(time=[time_coord])
    
    return result_dataset


def multiply_variables(dataset, variables=None, new_variable_name=None, time=None):
    
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif isinstance(dataset, xr.Dataset):
        dataset = dataset
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")   
    
    # Ensure all specified variables are in the dataset
    for var in variables:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    if variables is None:
        exclude_vars = ("grid_area", "land_frac")
        variables = [var for var in dataset.data_vars if not var.startswith(exclude_vars)]
    
    # Ensure all specified variables are in the dataset
    for var in variables:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    # Fill NaNs with one before multiplying
    filled_vars = [dataset[var].fillna(0) for var in variables]
    
    # Multiply the specified variables
    product_data = filled_vars[0]
    for var in filled_vars[1:]:
        product_data *= var
    
    # Convert resulting ones back to NaNs
    product_data = product_data.where(product_data != 0, other=np.nan)
    
    if new_variable_name:
        # Create a new dataset with the resulting variable
        product_dataset = xr.Dataset({new_variable_name: product_data})
    else:
        product_dataset = xr.Dataset({'product_variable': product_data})
    
    if time is not None:
        time_coord = pd.to_datetime(time)
        product_dataset = product_dataset.expand_dims(time=[time_coord])
    
    return product_dataset


def average_variables(dataset, variables=None, new_variable_name=None, time=None):
    
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif isinstance(dataset, xr.Dataset):
        dataset = dataset
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")   
    
    # Ensure all specified variables are in the dataset
    for var in variables:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    if variables is None:
        exclude_vars = ("grid_area", "land_frac")
        variables = [var for var in dataset.data_vars if not var.startswith(exclude_vars)]
    
    # Ensure all specified variables are in the dataset
    for var in variables:
        if var not in dataset:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    # Fill NaNs with zero before averaging
    filled_vars = [dataset[var].fillna(0) for var in variables]
    
    # Calculate the average of the specified variables
    averaged_data = sum(filled_vars) / len(filled_vars)
    
    # Convert resulting zeros back to NaNs
    averaged_data = averaged_data.where(averaged_data != 0, other=np.nan)
    
    if new_variable_name:
        # Create a new dataset with the averaged variable
        averaged_dataset = xr.Dataset({new_variable_name: averaged_data})
    else:
        averaged_dataset = xr.Dataset({'averaged_variable': averaged_data})
    
    if time is not None:
        time_coord = pd.to_datetime(time)
        averaged_dataset = averaged_dataset.expand_dims(time=[time_coord])
    
    return averaged_dataset


def equal_interval(data, num_classes):
    min_val = data.min()
    max_val = data.max()
    interval = (max_val - min_val) / num_classes
    intervals = [min_val + i * interval for i in range(num_classes + 1)]
    intervals[-1] += 1e-10  # Extend the last bin a tiny bit to ensure the max value is included
    return pd.cut(data, bins=intervals, right=True, include_lowest=True, labels=False)

def quantile_classes(data, num_classes):
    # Remove NaN values from data
    data_clean = data.dropna()
    # Calculate quantiles
    quantiles = np.linspace(0, 1, num_classes + 1)
    return pd.qcut(data_clean, q=quantiles, labels=False, precision=2)

def geometric_interval(data, num_classes):
    min_val = data.min()
    max_val = data.max()
    if min_val <= 0:
        min_val = 1  # adjust because geometric progression cannot start at 0 or negative
    ratio = (max_val / min_val) ** (1 / num_classes)
    intervals = [min_val * (ratio ** i) for i in range(num_classes + 1)]
    # Extend the last interval slightly to ensure max value is included
    intervals[-1] = max_val * (1 + 1e-10)  # Slightly extend the last bin
    return pd.cut(data, bins=intervals, include_lowest=True, labels=False)

def standard_deviation(data, num_classes, width=1):
    mean_val = data.mean()
    std_dev = data.std()
    # Calculate the range of standard deviations to cover
    intervals = [mean_val + i * width * std_dev for i in range(-num_classes // 2, num_classes // 2 + 1)]
    
    # Extend the intervals to include all data
    min_val = data.min()
    max_val = data.max()
    if min_val < intervals[0]:
        intervals.insert(0, min_val)  # Insert minimum value if it's outside the first interval
    if max_val > intervals[-1]:
        intervals.append(max_val)  # Append maximum value if it's outside the last interval
    
    return pd.cut(data, bins=intervals, include_lowest=True, labels=False)


