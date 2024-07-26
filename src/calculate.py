import os
import re
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import pyproj
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def calculate_geometry_attributes(input_gdf):
    gdf = input_gdf.copy()

    # Ensure the coordinate system is WGS84
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=4326)

    # Initialize the WGS84 ellipsoid
    geod = pyproj.Geod(ellps="WGS84")

    # Calculate area and length for each feature
    areas = []
    lengths = []

    for geom in gdf.geometry:
        if geom.is_valid:  # Check if the geometry is valid
            if isinstance(geom, Polygon):
                # Calculate the geodesic area for polygons
                area = abs(geod.geometry_area_perimeter(geom)[0])
                areas.append(np.float64(area))  # Convert to double precision
                lengths.append(None)  # No length for polygons
            elif isinstance(geom, (LineString, MultiLineString)):  # Handle MultiLineString geometries
                # Calculate the geodesic length for lines
                length = geod.geometry_length(geom)
                lengths.append(np.float64(length))  # Convert to double precision
                areas.append(None)  # No area for lines
            else:
                areas.append(None)
                lengths.append(None)
        else:
            areas.append(None)
            lengths.append(None)

    # Add the new attributes to the GeoDataFrame
    gdf['area_m2'] = areas
    gdf['length_m'] = lengths

    # Remove columns with all None values
    if gdf['area_m2'].isnull().all():
        gdf.drop(columns=['area_m2'], inplace=True)
    if gdf['length_m'].isnull().all():
        gdf.drop(columns=['length_m'], inplace=True)

    return gdf

def calculate_geometry_attributes(input_gdf):
    gdf = input_gdf.copy()

    # Ensure the coordinate system is WGS84
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Initialize the WGS84 ellipsoid
    geod = pyproj.Geod(ellps="WGS84")

    # Calculate area and length for each feature
    areas = []
    lengths = []

    for idx, geom in enumerate(gdf.geometry):
        if geom.is_valid:  # Check if the geometry is valid
            if isinstance(geom, (Polygon, MultiPolygon)):
                # Calculate the geodesic area for polygons
                area = abs(geod.geometry_area_perimeter(geom)[0])
                areas.append(np.float64(area))  # Convert to double precision
                lengths.append(None)  # No length for polygons
            elif isinstance(geom, (LineString, MultiLineString)):
                # Calculate the geodesic length for lines
                length = geod.geometry_length(geom)
                lengths.append(np.float64(length))  # Convert to double precision
                areas.append(None)  # No area for lines
            else:
                areas.append(None)
                lengths.append(None)
        else:
            try:
                # Attempt to fix invalid geometry with buffer(0)
                fixed_geom = geom.buffer(0)
                if fixed_geom.is_valid:
                    if isinstance(fixed_geom, (Polygon, MultiPolygon)):
                        # Calculate the geodesic area for polygons
                        area = abs(geod.geometry_area_perimeter(fixed_geom)[0])
                        areas.append(np.float64(area))  # Convert to double precision
                        lengths.append(None)  # No length for polygons
                    elif isinstance(fixed_geom, (LineString, MultiLineString)):
                        # Calculate the geodesic length for lines
                        length = geod.geometry_length(fixed_geom)
                        lengths.append(np.float64(length))  # Convert to double precision
                        areas.append(None)  # No area for lines
                    else:
                        areas.append(None)
                        lengths.append(None)
                else:
                    raise ValueError("Geometry could not be fixed")
            except Exception as e:
                # Print the row if geometry is not valid and could not be fixed
                print(f"Invalid geometry at index {idx} and could not be fixed: {geom}")
                print(gdf.iloc[idx])
                print(f"Error: {e}")
                areas.append(None)
                lengths.append(None)

    # Add the new attributes to the GeoDataFrame
    gdf['area_m2'] = areas
    gdf['length_m'] = lengths

    # Remove columns with all None values
    if gdf['area_m2'].isnull().all():
        gdf.drop(columns=['area_m2'], inplace=True)
    if gdf['length_m'].isnull().all():
        gdf.drop(columns=['length_m'], inplace=True)

    return gdf


import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import pyproj
import numpy as np

def calculate_geometry_attributes(input_gdf, column_name=None):
    """
    Calculate area or length for each geometry in the GeoDataFrame and store it in a specified column.

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

    # Ensure the coordinate system is WGS84
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Initialize the WGS84 ellipsoid
    geod = pyproj.Geod(ellps="WGS84")

    # Initialize list to store the calculated values
    values = []

    # Determine the appropriate column name if not provided
    if column_name is None:
        # Default to 'area_m2' or 'length_m' based on geometry type
        # Check if there are any polygons or lines to determine column name
        if any(isinstance(geom, (Polygon, MultiPolygon)) for geom in gdf.geometry):
            column_name = 'area_m2'
        elif any(isinstance(geom, (LineString, MultiLineString)) for geom in gdf.geometry):
            column_name = 'length_m'
        else:
            column_name = 'geometry_value'  # Default to a generic name if no polygons or lines are found

    # Calculate area or length for each geometry
    for geom in gdf.geometry:
        if geom.is_valid:
            if isinstance(geom, (Polygon, MultiPolygon)):
                # Calculate the geodesic area for polygons
                area = abs(geod.geometry_area_perimeter(geom)[0])
                values.append(np.float64(area))
            elif isinstance(geom, (LineString, MultiLineString)):
                # Calculate the geodesic length for lines
                length = geod.geometry_length(geom)
                values.append(np.float64(length))
            else:
                values.append(None)
        else:
            try:
                # Attempt to fix invalid geometry with buffer(0)
                fixed_geom = geom.buffer(0)
                if fixed_geom.is_valid:
                    if isinstance(fixed_geom, (Polygon, MultiPolygon)):
                        area = abs(geod.geometry_area_perimeter(fixed_geom)[0])
                        values.append(np.float64(area))
                    elif isinstance(fixed_geom, (LineString, MultiLineString)):
                        length = geod.geometry_length(fixed_geom)
                        values.append(np.float64(length))
                    else:
                        values.append(None)
                else:
                    raise ValueError("Fixed geometry is still invalid")
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
        

def sum_variables(variables=None, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.load_dataset(netcdf_directory)
    
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    if variables is None:
        variables = [var for var in list(dataset.data_vars) if not var.startswith("grid_area")]
    
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



def subtract_variables(variable1, variable2, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.load_dataset(netcdf_directory)
        
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    # Ensure both specified variables are in the dataset
    if variable1 not in dataset or variable2 not in dataset:
        raise ValueError(f"Both variables '{variable1}' and '{variable2}' must be present in the dataset.")
    
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


def divide_variables(variable1, variable2, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")

    if netcdf_directory:
        dataset = xr.load_dataset(netcdf_directory)
        
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    # Ensure both specified variables are in the dataset
    if variable1 not in dataset or variable2 not in dataset:
        raise ValueError(f"Both variables '{variable1}' and '{variable2}' must be present in the dataset.")
    
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


def multiply_variables(variables=None, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
        
    if netcdf_directory:
        dataset = xr.load_dataset(netcdf_directory)
        
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
    
    if variables is None:
        variables = [var for var in list(dataset.data_vars) if not var.startswith("grid_area")]
    
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


def average_variables(variables=None, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
   
    if netcdf_directory:
        dataset = xr.load_dataset(netcdf_directory)
        
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest')
        
    if variables is None:
        variables = [var for var in list(dataset.data_vars) if not var.startswith("grid_area")]
    
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

