import os
import re
import geopandas as gpd
import pandas as pd
import sys
from shapely.geometry import Polygon, LineString, Point
import pyproj
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
from shapely.geometry import box

# from . import create
# from . import calculate
# from . import get

import create
import calculate
import get


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# warning_message = (
#     "The land fraction data is not available at the resolution you requested, which prevents taking into account the portion of coastal grid cells that are not land.\n"
#     "Instead, the calculated values will be referenced to the total grid area per cell."
# )

warning_message = (
    "The land fraction data is not available at the resolution you requested. Only grid_area variables is added to the final dataset."
)

global_attr = {'Project': 'Surface Earth System Analysis and Modeling Environment (SESAME)',
               'Research Group': 'Integrated Earth System Dynamics',
               'Institution': 'McGill University',
               'Contact': 'eric.galbraith@mcgill.ca',
               'Data Version': 'V1.0'}

def replace_special_characters(value):
    """
    Replace special characters in a string with underscores and clean up consecutive underscores.

    Parameters:
    -----------
    value : str
        Input string containing special characters.

    Returns:
    --------
    cleaned_value : str
        Cleaned string with special characters replaced by underscores and consecutive underscores removed.
    """

    value = re.sub(r'[^\w]', '_', value)
    cleaned_value = re.sub(r'[_\s]+', '_', value)
    cleaned_value = cleaned_value.lower()

    return cleaned_value


def reverse_replace_special_characters(value):
    """
    Replace underscores with white spaces and capitalize each word.

    Parameters:
    -----------
    value : str
        Input string containing underscored characters.

    Returns:
    --------
    reversed_value : str
        Cleaned string with capitalized words.
    """
    parts = value.split('_')
    capitalized_parts = [part.capitalize() for part in parts]
    reversed_value = ' '.join(capitalized_parts)
    return reversed_value



def adjust_points(points_gdf, polygons_gdf, x_offset=0.0001, y_offset=0.0001):
    adjusted_points = []
    # Create the spatial index for the polygons
    sindex = polygons_gdf.sindex

    for idx, point in points_gdf.iterrows():
        adjusted_point = point['geometry']
        # Create a bounding box around the point
        bbox = point['geometry'].buffer(1e-14).bounds
        # Get the indices of the polygons that intersect with the bounding box
        possible_matches_index = list(sindex.intersection(bbox))
        # Get the corresponding polygons
        possible_matches = polygons_gdf.iloc[possible_matches_index]
        # Check if the point is within any of the possible match polygons
        point_within_polygon = possible_matches.contains(point['geometry']).any()
        if not point_within_polygon:
            adjusted_point = Point(point['geometry'].x + x_offset, point['geometry'].y + y_offset)
        adjusted_points.append(adjusted_point)
    
    points_gdf['geometry'] = adjusted_points
    return points_gdf


def add_variable_attributes(ds, variable_name, long_name, units, source=None, time=None, zero_is_value=False):
    """
    Adds attributes to a variable in an xarray Dataset.
    
    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variable.
    - variable_name: str
        The name of the variable to add attributes to.
    - long_name: str
        The long name of the variable.
    - units: str
        The units of the variable.
    - source: str, optional
        The source of the data.
        
    Returns:
    - ds: xarray.Dataset
        The dataset with updated variable attributes.
    """
    if variable_name not in ds:
        raise ValueError(f"Variable '{variable_name}' not found in the dataset.")
    
    ## add lat and lon attributes
    lon_attr = {"units" : "degrees_east",
            "modulo" : 360.,
            "point_spacing" : "even",
            "axis" : "X"}

    lat_attr = {"units" : "degrees_north",
                "point_spacing" : "even",
                "axis" : "Y"}


    ds['lat'].attrs = lat_attr
    ds['lon'].attrs = lon_attr
    
    # Add time dimension if provided
    if time is not None:
        time_d = pd.to_datetime(time)
        ds = ds.assign_coords(time=time_d)
        ds = ds.expand_dims(dim='time')
        time_attr = {
            "standard_name": "time",
            "axis": "T"}
        ds['time'].attrs = time_attr

    if zero_is_value:
        ds = ds
    else:
        # Replace the 0 values to NaN, where zero shows evidence of absence.
        ds[variable_name] = ds[variable_name].where(ds[variable_name] != 0, np.nan)
        
    # Add variable metadata
    attrs = {'long_name': long_name, 'units': units}
    if source is not None:
        attrs['source'] = source
        
    ds[variable_name].attrs = attrs
    # Set and add global attributes
    ds.attrs = global_attr
    
    return ds

def gridded_poly_2_dataset(polygon_gdf, grid_value, resolution, variable_name=None):

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Ensure the coordinate system is WGS84
    polygon_gdf.set_crs(epsg=4326, inplace=True)
    polygon_gdf = polygon_gdf.to_crs(epsg=4326)

    # Extract latitudes and longitudes from the geometry column
    polygon_gdf['lon'] = polygon_gdf['geometry'].centroid.x
    polygon_gdf['lat'] = polygon_gdf['geometry'].centroid.y

    num_lon_points = int(360 / resolution)
    num_lat_points = int(180 / resolution)

    lons = np.linspace(-180 + resolution/2, 180 - resolution/2, num_lon_points)
    lats = np.linspace(-90 + resolution/2, 90 - resolution/2, num_lat_points)

    # Create a meshgrid of coordinates
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Create a mask using the GeoDataFrame
    mask = np.zeros_like(lon_mesh, dtype=np.float64)
    
    # Iterate through each row in the GeoDataFrame
    for idx, row in polygon_gdf.iterrows():
        lon_idx = np.where(lons == row['lon'])[0]
        lat_idx = np.where(lats == row['lat'])[0]
        mask[lat_idx, lon_idx] = row[grid_value] 
    
    if variable_name:
        variable_name = replace_special_characters(variable_name)
        # Create an xarray dataset
        ds = xr.Dataset({
                variable_name: (['lat', 'lon'], mask)},
                coords={'lat': (['lat'], lats),
                'lon': (['lon'], lons)})
    else:
        grid_value = replace_special_characters(grid_value)
        ds = xr.Dataset({
                grid_value: (['lat', 'lon'], mask)},
                coords={'lat': (['lat'], lats),
                'lon': (['lon'], lons)})
    return ds

def add_grid_variables(ds, resolution, variable_name, normalize_by_area):
    
    # Ignore FutureWarning for other functions
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Ensure UserWarning is always shown during this function
    warnings.simplefilter('always', UserWarning)
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")     
    resolution_str = str(resolution)
    if resolution_str == "1" or resolution_str == "1.0":
        # Add grid area variable  
        grid_ds = xr.open_dataset(os.path.join(data_dir, "G.land_sea_mask.1deg.nc"))
        # Merge with the dataset
        ds = xr.merge([ds, grid_ds])       
        if normalize_by_area:
            ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    
    elif resolution_str == "0.5":       
        grid_ds = xr.open_dataset(os.path.join(data_dir, "G.land_sea_mask.0_5deg.nc"))
        # Merge with the dataset
        ds = xr.merge([ds, grid_ds])       
        if normalize_by_area:
            ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
            
    elif resolution_str == "0.25":          
        grid_ds = xr.open_dataset(os.path.join(data_dir, "G.land_sea_mask.0_25deg.nc"))
        # Merge with the dataset
        ds = xr.merge([ds, grid_ds])       
        if normalize_by_area:
            ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    else:
        gdf = create.create_gridded_polygon(resolution=resolution, grid_area="yes")
        grid_ds = gridded_poly_2_dataset(polygon_gdf=gdf, grid_value="grid_area", resolution=resolution)
        attrs = {'long_name': "Area of Grids", 'units': "m2"}
        grid_ds["grid_area"].attrs = attrs
        ds = xr.merge([ds, grid_ds])
        if normalize_by_area:
            # Issue a warning if land fraction data is not available at the desired resolution
            warnings.warn(warning_message, UserWarning)
            ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    
    # Restore the warning filter to its default state
    warnings.simplefilter('default', UserWarning)

    return ds

def gridded_poly_2_xarray(polygon_gdf, grid_value, long_name, units, resolution, source=None, time=None, variable_name=None, normalize_by_area=False, zero_is_value=False):

    ds = gridded_poly_2_dataset(polygon_gdf=polygon_gdf, grid_value=grid_value, variable_name=variable_name, resolution=resolution)
    variable_name = replace_special_characters(variable_name)

    ds = add_grid_variables(ds=ds, resolution=resolution, variable_name=variable_name, normalize_by_area=normalize_by_area)
    
    # merge with the dataset
    variable_name = variable_name if variable_name else grid_value
    ds = add_variable_attributes(ds=ds, variable_name=variable_name, long_name=long_name, units=units, source=source, time=time, zero_is_value=zero_is_value)

    return ds

def da_to_ds(da, variable_name, long_name, units, source=None, time=None, resolution=1, zero_is_value=False, normalize_by_area=False):
    """
    Convert a DataArray to a Dataset including additional metadata and attributes.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray to be converted.
    short_name : str
        Name of the variable.
    long_name : str
        A long name for the variable.
    units : str
        Units of the variable.
    source : str, optional
        Source information, if available. Default is None.
    time : str or None, optional
        Time information. If provided and not 'recent', a time dimension is added to the dataset.
        Default is None.
    resolution : float, optional
        Grid cell size for latitude and longitude bounds calculations. Default is 1.

    Returns
    -------
    xarray.Dataset
        The converted Dataset with added coordinates, attributes, and global attributes.
    """

    # Convert the DataArray to a Dataset with the specified variable name
    ds = da.to_dataset(name=variable_name)

    ## add lat and lon attributes
    lon_attr = {"units" : "degrees_east",
            "modulo" : 360.,
            "point_spacing" : "even",
            "axis" : "X"}

    lat_attr = {"units" : "degrees_north",
                "point_spacing" : "even",
                "axis" : "Y"}


    ds['lat'].attrs = lat_attr
    ds['lon'].attrs = lon_attr
    
    # Add time dimension if provided
    if time is not None:
        time_d = pd.to_datetime(time)
        ds = ds.assign_coords(time=time_d)
        ds = ds.expand_dims(dim='time')
        time_attr = {
            "standard_name": "time",
            "axis": "T"}
        ds['time'].attrs = time_attr

    if zero_is_value:
        ds = ds
    else:
        # Replace the 0 values to NaN, where zero shows evidence of absence.
        ds[variable_name] = ds[variable_name].where(ds[variable_name] != 0, np.nan)

    # add grid area variable
    resolution = abs(float(ds['lat'].diff('lat').values[0]))
    ds = add_grid_variables(ds=ds, resolution=resolution, variable_name=variable_name, normalize_by_area=normalize_by_area)

    # Add variable metadata
    attrs = {'long_name': long_name, 'units': units}
    if source is not None:
        attrs['source'] = source
        
    ds[variable_name].attrs = attrs
    # Set and add global attributes
    ds.attrs = global_attr
    
    return ds

def determine_long_name_point(agg_column, variable_name, long_name, agg_function):
    if long_name is None:
        if agg_column is None or (agg_function is not None and agg_function.lower() == 'sum'):
            return reverse_replace_special_characters(variable_name) if variable_name else "count"
    return long_name if long_name else "count"


def determine_units_line(units, normalize_by_area):
    if units == "meter/grid-cell" and normalize_by_area:
        return "m m-2"
    return units

def determine_long_name_line(agg_column, variable_name, long_name, agg_function):
    if long_name is None:
        if agg_column is None or (agg_function is not None and agg_function.lower() == 'sum'):
            return reverse_replace_special_characters(variable_name) if variable_name else reverse_replace_special_characters(f"length_{agg_function.lower()}")
    return long_name if long_name else reverse_replace_special_characters(f"length_{agg_function.lower()}")

def determine_long_name_line(long_name, agg_column, variable_name):
    if long_name is None:
        if variable_name:
            long_name = reverse_replace_special_characters(variable_name)
        else:
            long_name = reverse_replace_special_characters(agg_column)
    return long_name
        

def dataframe_stats_line(dataframe, agg_column=None, agg_function="sum"):
    if agg_function.lower() == "sum":
        if agg_column is None:
            global_summary_stats = dataframe[variable_name].sum()
            return global_summary_stats
        else:
            variable_name = agg_column or "length_m"
            dataframe = calculate.calculate_geometry_attributes(dataframe)
            global_summary_stats = dataframe["length_m"].sum()
            return global_summary_stats * 1e-3
    else:
        raise ValueError(f"Unsupported agg_function: {agg_function}. Choose 'sum'. or set verbose=False")
    
def determine_units_poly(units, normalize_by_area, fraction):
    if fraction:
        if fraction and not normalize_by_area:
            return "fraction"
        if fraction and normalize_by_area:
            raise ValueError("Fraction and value per area cannot be created together.")
    elif normalize_by_area:
        units = "m2 m-2"
    return units
    

def determine_long_name_poly(variable_name, long_name, agg_function):
    if long_name is None:
        if variable_name is None or (agg_function is not None and agg_function.lower() == 'sum'):
            return reverse_replace_special_characters(variable_name) if variable_name else "area"
    return long_name if long_name else "area"


def dataframe_stats_poly(dataframe, agg_function="sum"):
    if agg_function.lower() == "sum":
        dataframe = calculate.calculate_geometry_attributes(dataframe)
        global_summary_stats = dataframe['area_m2'].sum()
        return global_summary_stats * 1e-6
    else:
        raise ValueError(f"Unsupported agg_function: {agg_function}. Choose 'sum'. or set verbose=False")
    

def determine_units_point(units, normalize_by_area):
    if units == "value/grid-cell" and normalize_by_area:
        return "value m-2"
    return units


def dataframe_stats_point(dataframe, agg_column=None, agg_function="sum"):
    if agg_column is None or agg_column == "count":
        if agg_function is None or agg_function.lower() == 'sum':
            global_summary_stats = len(dataframe)
        else:
            raise ValueError(f"Unsupported agg_function: {agg_function}")
    elif agg_function is not None and agg_function.lower() == 'sum' and agg_column is not None:
        global_summary_stats = dataframe[agg_column].sum()
    else:
        raise ValueError(f"Unsupported combination of agg_column: {agg_column} and agg_function: {agg_function}")
    return global_summary_stats


def xarray_dataset_stats(dataset, variable_name=None, agg_column=None, normalize_by_area=None, resolution=1):
    if variable_name is None and agg_column:
        variable_name = agg_column
    elif variable_name and agg_column:
        variable_name = variable_name
    if normalize_by_area:
        global_gridded_stats = (dataset[variable_name].fillna(0) * dataset["grid_area"]).sum().item()
    else:
        global_gridded_stats = (dataset[variable_name]).sum().item()
    return global_gridded_stats


def save_to_nc(ds, output_directory=None, output_filename=None, base_filename=None):
    if output_directory != None:
        if output_filename != None:
            ds.to_netcdf(output_directory + output_filename + ".nc")
        else:
            ds.to_netcdf(output_directory + base_filename + ".nc")
            
def point_spatial_join(polygons_gdf, points_gdf, agg_column=None, agg_function='sum', x_offset=0.0001, y_offset=0.0001):
    # Adjust points to ensure they are within or correctly intersecting polygons
    points_gdf = adjust_points(points_gdf, polygons_gdf, x_offset, y_offset)
    
    # Perform spatial join with 'intersects' operation
    points_within_polygons = gpd.sjoin(points_gdf, polygons_gdf, predicate='intersects')
    
    # If field_name is provided, convert the column to numeric
    if agg_column:
        points_within_polygons[agg_column] = pd.to_numeric(points_within_polygons[agg_column], errors='coerce')
        variable_name = replace_special_characters(agg_column)
        # Group by polygon and compute summary statistic based on operation
        if agg_function.lower() == 'sum':
            summary_stats = points_within_polygons.groupby('index_right')[agg_column].sum().reset_index(name=variable_name)
        elif agg_function.lower() == 'mean':
            summary_stats = points_within_polygons.groupby('index_right')[agg_column].mean().reset_index(name=variable_name)
        elif agg_function.lower() == 'max':
            summary_stats = points_within_polygons.groupby('index_right')[agg_column].max().reset_index(name=variable_name)
        elif agg_function.lower() == 'min':
            summary_stats = points_within_polygons.groupby('index_right')[agg_column].min().reset_index(name=variable_name)            
        elif agg_function.lower() == 'std':
            summary_stats = points_within_polygons.groupby('index_right')[agg_column].std().reset_index(name=variable_name)
        else:
            raise ValueError(f"Unsupported operation: {agg_column}. Choose from 'sum', 'mean', 'max', 'std'.")
    
        # Merge summary statistics with polygons GeoDataFrame
        polygons_gdf = polygons_gdf.merge(summary_stats, how='left', left_index=True, right_on='index_right')
        print(polygons_gdf[variable_name].sum())
    
    else:
        # Count the points within each polygon
        polygon_counts = points_within_polygons.groupby('index_right').size().reset_index(name='count')
        # Add the count to the polygons GeoDataFrame
        polygons_gdf['count'] = polygons_gdf.index.to_series().map(polygon_counts.set_index('index_right')['count']).fillna(0).astype(int)
    
    return polygons_gdf

def point_spatial_join(polygons_gdf, points_gdf, agg_column=None, agg_function='sum', x_offset=0.0001, y_offset=0.0001):
    # Ensure both GeoDataFrames use the same CRS
    if polygons_gdf.crs != points_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
        
    # Adjust points to ensure they are within or correctly intersecting polygons
    points_gdf = adjust_points(points_gdf, polygons_gdf, x_offset, y_offset)
    
    # Perform intersection
    intersections = gpd.overlay(points_gdf, polygons_gdf, how='intersection')
    
    # If agg_column is provided, convert the column to numeric
    if agg_column:
        intersections[agg_column] = pd.to_numeric(intersections[agg_column], errors='coerce')
        # Group by polygon and compute summary statistic based on operation
        if agg_function.lower() == 'sum':
            intersections = intersections.groupby('uid')[agg_column].sum().reset_index()
        elif agg_function.lower() == 'mean':
            intersections = intersections.groupby('uid')[agg_column].mean().reset_index()
        elif agg_function.lower() == 'max':
            intersections = intersections.groupby('uid')[agg_column].max().reset_index()
        elif agg_function.lower() == 'min':
            intersections = intersections.groupby('uid')[agg_column].min().reset_index()
        elif agg_function.lower() == 'std':
            intersections = intersections.groupby('uid')[agg_column].std().reset_index()
        else:
            raise ValueError(f"Unsupported operation: {agg_column}. Choose from 'sum', 'mean', 'max', min, 'std'.")

    else:
        intersections = intersections.groupby('uid').size().reset_index(name='count')
    
    joined_gdf = polygons_gdf.merge(intersections, on='uid', how='left')
    return joined_gdf
    

def line_intersect(polygons_gdf, lines_gdf, agg_column=None, agg_function='sum'):
    
    # Ensure both GeoDataFrames use the same CRS
    if polygons_gdf.crs != lines_gdf.crs:
        lines_gdf = lines_gdf.to_crs(polygons_gdf.crs)
    
    # Calculate geometry attributes
    polygons_gdf = calculate.calculate_geometry_attributes(input_gdf=polygons_gdf, column_name="grid_area")
    
    # Perform intersection
    intersections = gpd.overlay(lines_gdf, polygons_gdf, how='intersection', keep_geom_type=True)
    
    # Calculate geometry attributes for intersections
    intersections = calculate.calculate_geometry_attributes(input_gdf=intersections, column_name="length_m")

    # If agg_column is provided, convert the column to numeric
    if agg_column:
        intersections[agg_column] = pd.to_numeric(intersections[agg_column], errors='coerce')
        # variable_name = replace_special_characters(agg_column)
        # Group by polygon and compute summary statistic based on operation
        if agg_function.lower() == 'sum':
            intersections = intersections.groupby('uid')[agg_column].sum().reset_index()
        elif agg_function.lower() == 'mean':
            intersections = intersections.groupby('uid')[agg_column].mean().reset_index()
        elif agg_function.lower() == 'max':
            intersections = intersections.groupby('uid')[agg_column].max().reset_index()
        elif agg_function.lower() == 'min':
            intersections = intersections.groupby('uid')[agg_column].min().reset_index()
        elif agg_function.lower() == 'std':
            intersections = intersections.groupby('uid')[agg_column].std().reset_index()
        else:
            raise ValueError(f"Unsupported operation: {agg_column}. Choose from 'sum', 'mean', 'max', min, 'std'.")

    else:
        if agg_function.lower() == "sum":
            intersections = intersections.groupby('uid')['length_m'].sum().reset_index()
        elif agg_function.lower() == "mean":
            intersections = intersections.groupby('uid')['length_m'].mean().reset_index()
        elif agg_function.lower() == "max":
            intersections = intersections.groupby('uid')['length_m'].max().reset_index()
        elif agg_function.lower() == "min":
            intersections = intersections.groupby('uid')['length_m'].min().reset_index()
        elif agg_function.lower() == "std":
            intersections = intersections.groupby('uid')['length_m'].std().reset_index()
    
    joined_gdf = polygons_gdf.merge(intersections, on='uid', how='left')
    return joined_gdf


def convert_xarray_to_gdf(ds, variable_name, resolution=1):
    # Extract the data variable as a NumPy array
    data = ds[variable_name].values
    
    # Get the coordinates from the dataset
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    # Create a list to hold the polygons and their corresponding values
    polygons = []
    values = []
    
    # Loop through the data to create polygons
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Define the corners of the polygon centered around (lons[j], lats[i])
            polygon = Polygon([
                (lons[j] - resolution / 2, lats[i] - resolution / 2),  # Bottom left corner
                (lons[j] + resolution / 2, lats[i] - resolution / 2),  # Bottom right corner
                (lons[j] + resolution / 2, lats[i] + resolution / 2),  # Top right corner
                (lons[j] - resolution / 2, lats[i] + resolution / 2)   # Top left corner
            ])
            polygons.append(polygon)
            values.append(data[i, j])  # Use the corresponding value
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'grid_area': values  # Add the values as a column
    }, crs='EPSG:4326')  # Set the coordinate reference system

    return gdf


def poly_fraction(ds, variable_name, resolution, polygons_gdf=None):

    # Ignore FutureWarning for other functions
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Ensure UserWarning is always shown during this function
    warnings.simplefilter('always', UserWarning)

    variable_name = replace_special_characters(variable_name)
    # Save the attributes of the variable
    attrs = ds[variable_name].attrs
    
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")
    resolution_str = str(resolution)
    if resolution_str == "1" or resolution_str == "1.0":    
        grid_ds = xr.open_dataset(os.path.join(data_dir, "G.land_sea_mask.1deg.nc"))
        ds = xr.merge([ds, grid_ds])
        ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]

    elif resolution_str == "0.5":     
        grid_ds = xr.open_dataset(os.path.join(data_dir, "G.land_sea_mask.0_5deg.nc"))
        # Merge with the dataset
        ds = xr.merge([ds, grid_ds])       
        ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    
    elif resolution_str == "0.25":    
        grid_ds = xr.open_dataset(os.path.join(data_dir, "G.land_sea_mask.0_25deg.nc"))
        # Merge with the dataset
        ds = xr.merge([ds, grid_ds])       
        ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]

    else:
        # Issue a warning if land fraction data is not available at the desired resolution
        warnings.warn(warning_message, UserWarning)
        grid_ds = gridded_poly_2_dataset(polygon_gdf=polygons_gdf, grid_value="grid_area", resolution=resolution)
        attrs = {'long_name': "Area of Grids", 'units': "m2"}
        grid_ds["grid_area"].attrs = attrs
        ds = xr.merge([ds, grid_ds])
        ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    
    # Reassign the saved attributes back to the variable
    ds[variable_name].attrs = attrs
    
    # Restore the warning filter to its default state
    warnings.simplefilter('default', UserWarning)
    return ds

def poly_intersect(poly_gdf, polygons_gdf, variable_name, long_name, units, source, time, resolution, agg_function="sum", normalize_by_area=None, zero_is_value=None, fraction=False):
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Calculate geometry attributes
    poly_gdf = calculate.calculate_geometry_attributes(input_gdf=poly_gdf, column_name="raw_area")
    polygons_gdf = calculate.calculate_geometry_attributes(input_gdf=polygons_gdf, column_name="grid_area")
    # Perform intersection
    intersections = gpd.overlay(poly_gdf, polygons_gdf, how='intersection')
    # Calculate geometry attributes for intersections
    intersections = calculate.calculate_geometry_attributes(input_gdf=intersections, column_name="in_area")

    if agg_function.lower() == "sum":
        intersections = intersections.groupby('uid')['in_area'].sum().reset_index()
    elif agg_function.lower() == "mean":
        intersections = intersections.groupby('uid')['in_area'].mean().reset_index()
    elif agg_function.lower() == "max":
        intersections = intersections.groupby('uid')['in_area'].max().reset_index()
    elif agg_function.lower() == "min":
        intersections = intersections.groupby('uid')['in_area'].min().reset_index()
    elif agg_function.lower() == "std":
        intersections = intersections.groupby('uid')['in_area'].std().reset_index()

    joined_gdf = polygons_gdf.merge(intersections, on='uid', how='left')

    ds = gridded_poly_2_xarray(
                polygon_gdf=joined_gdf,
                grid_value='in_area',
                long_name=long_name,
                units=units,
                source=source,
                time=time,
                resolution=resolution,
                variable_name=variable_name,
                normalize_by_area=normalize_by_area,
                zero_is_value=zero_is_value
            )

    if fraction:
        ds = poly_fraction(ds=ds, variable_name=variable_name, resolution=resolution, polygons_gdf=polygons_gdf)

    return ds



def netcdf_2_tif(raster_data, netcdf_variable, time=None):
    """
    Convert NetCDF data to a TIFF file. Handles multidimensional data and specific times.
    Ensures proper projection and cell size.

    Parameters
    ----------
    netcdf_path : str
        File path to the NetCDF data.
    netcdf_variable : str
        Variable in the NetCDF data to be converted.
    temp_path : str
        Temporary path to save the TIFF file.
    time : str or None, optional
        Specific time for the conversion in the format YYYY-MM-DD. Default is None.

    Returns
    -------
    str
        File path to the generated TIFF file.
    """
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(raster_data, (str, bytes, os.PathLike)):
        ds = xr.open_dataset(raster_data)
    elif isinstance(raster_data, xr.Dataset):
        ds = raster_data
    else:
        raise TypeError("`raster_data` must be an xarray.Dataset or a path to a NetCDF file.")
    # create a temp path
    
    temp_path = create.create_temp_folder(folder_name="temp")
    
    # Select the variable
    data = ds[netcdf_variable]
    
    # Handle multidimensional data and specific time
    if time is not None:
        data = data.sel(time=time, method="nearest").drop_vars("time")
    
    data = data.squeeze()  # Remove any singleton dimensions
    # Extract the data array
    array = data.values
    
    # identify lat, lon variables
    x_dimension, y_dimension = get.identify_lat_lon_names(raster_data)
    # Extract latitude and longitude from dataset
    lon = ds[x_dimension].values
    lat = ds[y_dimension].values
    
    # Shift the prime meridian to Greenwich if the raw longitude is defined otherwise
    if lon.min() >= 0 and lon.max() > 180:
        lon = lon - 180
        # Shift the data array to match the new longitude values
        shift_index = np.where(lon >= 0)[0][0]
        array = np.roll(array, shift_index, axis=1)
    
    # Get dimensions
    height, width = array.shape
    
    # Calculate transform based on extent and cell size
    min_lon, max_lon = lon.min(), lon.max()
    min_lat, max_lat = lat.min(), lat.max()

    if lat[0] < lat[-1]:
        lat = np.flip(lat)
        array = np.flipud(array)  # Flip the array vertically to match the lat reversal
    
    # transform = from_origin(min_lon, max_lat, (max_lon - min_lon) / width, (max_lat - min_lat) / height)
    lon_res = lon[1] - lon[0]
    lat_res = abs(lat[1] - lat[0])
    transform = from_origin(min_lon, max_lat, lon_res, lat_res)

    # Prepare for writing the TIFF file
    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': str(array.dtype),
        'width': width,
        'height': height,
        'crs': CRS.from_epsg(4326).to_wkt(),
        'transform': transform
    }
    
    # Generate the output TIFF path
    time_str = str(time)
    raster_layer = f"{netcdf_variable}_{time_str[:10] if time else ''}.tif"
    output_raster = os.path.join(temp_path, raster_layer)
    
    # Write the data to a TIFF file
    with rasterio.open(output_raster, 'w', **metadata) as dst:
        dst.write(array, 1)
    
    return output_raster, temp_path


def reproject_and_fill(input_raster, dst_extent=(-180.0, -90.0, 180.0, 90.0)):
    """
    Reproject the input raster to WGS84 projection and fill the missing rasters as 0.
    The output raster will also maintain the specified extent.

    Parameters
    ----------
    input_raster : str
        File path to the input raster.
    dst_extent : tuple, optional
        The desired output extent (minX, minY, maxX, maxY). Default is (-180.0, -90.0, 180.0, 90.0).

    Returns
    -------
    np.ndarray
        The reprojected data as a numpy array.
    float
        The size of the cell in x direction.
    float
        The size of the cell in y direction.
    """

    # Define the target CRS (WGS84)
    dst_crs = CRS.from_epsg(4326)

    # Open the input raster
    with rasterio.open(input_raster) as src:
        # Check if the CRS of the input raster is already WGS84
        if src.crs == dst_crs:
            # If the CRS is already WGS84, only maintain the extent
            src_crs = src.crs
            src_transform = src.transform
            src_res = (abs(src_transform[0]), abs(src_transform[4]))  # (pixel_width, pixel_height)

            # Calculate the dimensions of the output raster based on the extent and input resolution
            dst_width = round((dst_extent[2] - dst_extent[0]) / src_res[0])
            dst_height = round((dst_extent[3] - dst_extent[1]) / abs(src_res[1]))

            # Calculate the transform for the output raster
            dst_transform = from_bounds(*dst_extent, dst_width, dst_height)

            # Create an array to hold the reprojected data
            dst_array = np.full((dst_height, dst_width), 0, dtype=src.dtypes[0])

            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0
                )
        else:
            # If the CRS is not WGS84, reproject to WGS84
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            # Create an array to hold the reprojected data
            dst_array = np.full((height, width), 0, dtype=src.dtypes[0])

            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0
                )

            # If needed, reapply the extent based on the output of reprojection
            # Adjust the resolution to fit the new extent
            # src_res = (abs(src_transform[0]), abs(src_transform[4]))
            src_res = (abs(transform[0]), abs(transform[4]))
            dst_width = round((dst_extent[2] - dst_extent[0]) / src_res[0])
            dst_height = round((dst_extent[3] - dst_extent[1]) / abs(src_res[1]))
            dst_transform = from_bounds(*dst_extent, dst_width, dst_height)

            # Create a new array for the final extent
            final_array = np.full((dst_height, dst_width), 0, dtype=dst_array.dtype)

            # Reproject the already reprojected array to the new extent
            reproject(
                source=dst_array,
                destination=final_array,
                src_transform=transform,
                src_crs=dst_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                dst_nodata=0
            )
            dst_array = final_array

    # Calculate the cell size in x and y direction
    x_resolution = (dst_extent[2] - dst_extent[0]) / dst_width
    y_resolution = (dst_extent[3] - dst_extent[1]) / dst_height

    return dst_array, x_resolution, y_resolution


def regrid_array_2_ds(array, agg_function, variable_name, long_name, units="value/grid-cell", source=None, resolution=1,
                     time=None, zero_is_value=None, padding="symmetric", normalize_by_area=False, verbose=False):
    
    arr = array.astype(np.float64)

    # Calculate raw global value based on the conversion type
    if agg_function.upper() == 'SUM':
        raw_global_value = arr.sum()

    # Get dimensions and calculate grid resolution
    num_rows, num_cols = arr.shape
    num_lat, num_lon = calculate.calculate_grid_resolution(resolution=resolution)
    
    # Set a tolerance level
    tolerance = 0.01
    
    # Check if the dimensions are perfectly divisible within the tolerance
    if abs(num_rows / num_lat - round(num_rows / num_lat)) < tolerance and abs(num_cols / num_lon - round(num_cols / num_lon)) < tolerance:
        padded_arr = arr
    else:
        # Calculate padding needed for the array
        lat_padding = num_lat - (arr.shape[0] % num_lat)
        lon_padding = num_lon - (arr.shape[1] % num_lon)
        
        if padding.lower() == "end":
            # Distribute the padding only to the end of the array
            padded_arr = np.pad(arr, ((0, lat_padding), (0, lon_padding)), mode='constant', constant_values=0)
        else:
            # Distribute the padding evenly to the start and end of the array
            lat_padding_start = lat_padding // 2
            lat_padding_end = lat_padding - lat_padding_start
            lon_padding_start = lon_padding // 2
            lon_padding_end = lon_padding - lon_padding_start
            # Pad the array with zeros
            padded_arr = np.pad(arr, ((lat_padding_start, lat_padding_end), (lon_padding_start, lon_padding_end)), mode='constant', constant_values=0)
        

    # Calculate factors for latitude and longitude
    lat_factor = padded_arr.shape[0] // num_lat
    lon_factor = padded_arr.shape[1] // num_lon

    aligned_arr = np.zeros((num_lat, num_lon, lat_factor, lon_factor))

    for i in range(num_lat):
        for j in range(num_lon):
            lat_start = i * lat_factor
            lat_end = (i + 1) * lat_factor
            lon_start = j * lon_factor
            lon_end = (j + 1) * lon_factor
            aligned_arr[i, j] = padded_arr[lat_start:lat_end, lon_start:lon_end]

    lat_resolution = resolution
    lon_resolution = resolution

    lat = np.linspace(90 - lat_resolution / 2, -90 + lat_resolution / 2, num=num_lat)
    lon = np.linspace(-180 + lon_resolution / 2, 180 - lon_resolution / 2, num=num_lon)


    # Create an xarray DataArray with dimensions and coordinates
    da = xr.DataArray(aligned_arr, dims=("lat", "lon", "lat_factor", "lon_factor"), coords={"lat": lat, "lon": lon})

    # Perform the aggregation over the lat_factor and lon_factor dimensions
    if agg_function.upper() == 'SUM':
        da_agg = da.sum(dim=['lat_factor', 'lon_factor'])

    elif agg_function.upper() == 'MEAN':
        if zero_is_value:
            da_agg = da.mean(dim=['lat_factor', 'lon_factor'])
        else:
            # Calculate the sum and count of non-zero values
            da_sum = da.sum(dim=['lat_factor', 'lon_factor'])
            da_count = da.where(da != 0).count(dim=['lat_factor', 'lon_factor'])
            # Calculate the mean, handling division by zero
            da_agg_nan = da_sum / da_count.where(da_count != 0, np.nan)
            # Replace nan values with 0
            da_agg = da_agg_nan.fillna(0)
    elif agg_function.upper() == 'MAX':
        da_agg = da.max(dim=['lat_factor', 'lon_factor'])
    elif agg_function.upper() == 'MIN':
        da_agg = da.min(dim=['lat_factor', 'lon_factor'])
    elif agg_function.upper() == 'STD':
        da_agg = da.std(dim=['lat_factor', 'lon_factor'])
    else:
        raise ValueError("Conversion should be either SUM, MEAN, MAX, MIN or STD")

    if verbose and agg_function.upper() == 'SUM':
        print(f"Raw global {agg_function}: {raw_global_value:.3f}")
        regridded_global_value = da_agg.sum().item()
        print(f"Re-gridded global {agg_function}: {regridded_global_value:.3f}")
    
    # convert dataarray to dataset
    ds = da_to_ds(da_agg, variable_name, long_name, units, source, time, resolution, zero_is_value, normalize_by_area)

    return ds



def xy_not_eq(raster_path, agg_function, variable_name, long_name, units, source=None, time=None, resolution=1, 
                                 normalize_by_area=False, zero_is_value=False):
    # Convert raster to polygon GeoDataFrame
    gdf = raster_to_polygon_gdf(raster_file=raster_path)
    gdf = gdf.fillna(0)
    gdf = calculate.calculate_geometry_attributes(input_gdf=gdf, column_name="ras_area")
    # Create gridded polygon GeoDataFrame
    polygons_gdf = create.create_gridded_polygon(resolution=resolution)
    # Perform intersection
    intersections = gpd.overlay(gdf, polygons_gdf, how='intersection', keep_geom_type=True)
    # Calculate geometry attributes
    intersections = calculate.calculate_geometry_attributes(input_gdf=intersections, column_name="in_area")
    # Calculate the fraction
    intersections["frac"] = intersections["in_area"] / intersections["ras_area"]
    # compute statistics
    result_df = compute_weighted_statistics(gdf=intersections, stat=agg_function)
    # Merge results with polygons_gdf
    joined_gdf = polygons_gdf.merge(result_df, on='uid', how='left')
    variable_name = replace_special_characters(variable_name)
    ds = gridded_poly_2_xarray(polygon_gdf=joined_gdf, grid_value=agg_function, long_name=long_name, 
                                 units=units, source=source, time=time, variable_name=variable_name, resolution=resolution, 
                                 normalize_by_area=normalize_by_area, zero_is_value=zero_is_value)
    return ds



def raster_to_polygon_gdf(raster_file):
    # Open the raster file
    with rasterio.open(raster_file) as src:
        # Read the raster to an array
        array = src.read(1)
        
        # Handle nodata values
        if src.nodata is not None:
            mask = array != src.nodata
        else:
            mask = np.ones_like(array, dtype=bool)
        
        # Prepare lists for geometries and values
        geometries = []
        values = []
        
        # Loop through each pixel and create a polygon
        for i in range(src.height):
            for j in range(src.width):
                if mask[i, j]:
                    # Calculate the bounds of the pixel
                    left = src.transform * (j, i)
                    right = src.transform * (j + 1, i + 1)
                    polygon = box(left[0], left[1], right[0], right[1])
                    geometries.append(polygon)
                    values.append(array[i, j])
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': geometries, 'value': values})
        
        # Set the CRS of the GeoDataFrame to match the raster
        gdf.crs = src.crs
    
    return gdf

def compute_weighted_statistics(gdf, stat='sum'):
    if stat.lower() == 'sum':
        result = gdf.groupby('uid', as_index=False).apply(
            lambda df: pd.Series({
                'sum': (df['value'] * df['frac']).sum()
            }), include_groups=False
        ).reset_index()
    elif stat.lower() == 'mean':
        result = gdf.groupby('uid', as_index=False).apply(
            lambda df: pd.Series({
                'mean': (df['value'] * df['frac']).sum() / df['frac'].sum() if df['frac'].sum() != 0 else float('nan')
            }), include_groups=False
        ).reset_index()
    elif stat.lower() == 'max':
        result = gdf.groupby('uid', as_index=False).apply(
            lambda df: pd.Series({
                'max': (df['value'] * df['frac']).max()
            }), include_groups=False
        ).reset_index()
    elif stat.lower() == 'min':
        result = gdf.groupby('uid', as_index=False).apply(
            lambda df: pd.Series({
                'max': (df['value'] * df['frac']).min()
            }), include_groups=False
        ).reset_index()
    elif stat.lower() == 'std':
        result = gdf.groupby('uid', as_index=False).apply(
            lambda df: pd.Series({
                'std': (((df['value'] - ((df['value'] * df['frac']).sum() / df['frac'].sum() if df['frac'].sum() != 0 else float('nan'))) ** 2 * df['frac']).sum() / df['frac'].sum()) ** 0.5 if df['frac'].sum() != 0 else float('nan')
            }), include_groups=False
        ).reset_index()
    else:
        raise ValueError("Unsupported statistic specified. Choose from 'sum', 'mean', 'max', 'min, 'std'.")

    return result


def tif_2_ds(input_raster, variable_name, agg_function, long_name, units="value/grid-cell", source=None, resolution=1,
                     time=None, padding="symmetric", zero_is_value=False, normalize_by_area=False, verbose=False):
    
    # Step-1: Check the cell size
    # Open the input raster using rasterio
    with rasterio.open(input_raster) as raster:
        # Get the X and Y cell sizes from the raster properties
        x_size, y_size = raster.res[0], raster.res[1]

        # Round and convert cell sizes to float
        x_size = round(float(x_size), 3)
        y_size = round(float(y_size), 3)

        # Define a small tolerance for floating-point comparison
        tolerance = 1e-6

        # Check if x_size is approximately equal to y_size
        if abs(x_size - y_size) <= tolerance:
            # re-grid
            array, x_resolution, y_resolution = reproject_and_fill(input_raster)
            ds = regrid_array_2_ds(array=array, agg_function=agg_function, variable_name=variable_name, 
                                   long_name=long_name, units=units, source=source, resolution=resolution, 
                                   time=time, padding=padding, zero_is_value=zero_is_value, normalize_by_area=normalize_by_area, 
                                   verbose=verbose)
        else:
            ds = xy_not_eq(raster_path=input_raster, variable_name=variable_name, agg_function=agg_function, 
                           long_name=long_name, units=units, source=source, time=time, resolution=resolution, 
                           normalize_by_area=normalize_by_area, zero_is_value=zero_is_value)
    return ds


def adjust_datasets(input_ds, country_ds, time):
    # get the maximum time value of the netcdf file
    if any(var == 'time' for var in input_ds.variables):
        nc_max_time = input_ds.time.max().values
    else:
        nc_max_time = None

    if time == 'recent' or time is None:
        country_max_time = country_ds.time.max().values
        country_ds = country_ds.sel(time=country_max_time)
        if nc_max_time is not None:
            input_ds = input_ds.sel(time=nc_max_time)
        a = np.zeros((input_ds.dims['lat'], input_ds.dims['lon']), dtype=np.float64)
    elif nc_max_time is None and time is not None:
        country_ds = country_ds.sel(time=time)
        a = np.zeros((input_ds.dims['lat'], input_ds.dims['lon']), dtype=np.float64)
    else:
        country_ds = country_ds.sel(time=time)
        input_ds = input_ds.sel(time=time)
        a = np.zeros((input_ds.dims['lat'], input_ds.dims['lon']), dtype=np.float64)
    return input_ds, country_ds, a


def merge_ds_list(sd, dataset_list, netcdf_path=None, filename=None):
    ds = xr.merge(dataset_list)
    ds.attrs = {}  # Delete autogenerated global attributes
    ds.attrs.update(sd.global_attr)  # Adding new global attributes

    if netcdf_path is not None:
        ds.to_netcdf(netcdf_path + filename + ".nc")
    return ds

def delete_temporary_folder(folder_path):
    import shutil
    import os
    try:
        # Remove read-only attribute, if exists
        os.chmod(folder_path, 0o777)
        
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        # print(f"Successfully deleted the folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting the folder: {e}")


def grid_2_table(grid_data=None, variables=None, time=None, grid_area=False, resolution=1, aggregation=None, method='sum', verbose=False):
    
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")
    
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(grid_data, (str, bytes, os.PathLike)):
        ds = xr.open_dataset(grid_data)
    elif isinstance(grid_data, xr.Dataset):
        ds = grid_data
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")

    # Determine variables to process
    exclude_vars = ["time", "lat", "lon", "land_frac", "grid_area", "land_area"]
    if variables is not None:
        if isinstance(variables, str):
            variables_list = [variables]
        elif isinstance(variables, list):
            variables_list = variables
        else:
            raise ValueError("variables should be a string or a list of strings.")
    else:
        variables_list = [var for var in ds.data_vars if var not in exclude_vars]

    if verbose:
        print(f"List of variables to process: {variables_list}")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Load ISO3 to continent mapping from CSV
    try:
        iso3_continent_df = pd.read_csv(os.path.join(data_dir, "un_geoscheme.csv"), encoding='utf-8')
    except UnicodeDecodeError:
        iso3_continent_df = pd.read_csv(os.path.join(data_dir, "un_geoscheme.csv"), encoding='latin1')

    # Loop through each variable in the dataset
    for var in variables_list:
        try:
            resolution_str = str(resolution)
            if resolution_str == "1" or resolution_str == "1.0":
                cntry_ds = xr.open_dataset(os.path.join(data_dir, "country_fraction.1deg.2000-2023.a.nc"))
            elif resolution_str == "0.5":
                cntry_ds = xr.open_dataset(os.path.join(data_dir, "country_fraction.0_5deg.2000-2023.a.nc"))
            elif resolution_str == "0.25":
                cntry_ds = xr.open_dataset(os.path.join(data_dir, "country_fraction.0_25deg.2000-2023.a.nc"))
        except FileNotFoundError as e:
            print(f"Error while reading file {e}")

        # Handle the time dimension
        if 'time' in ds[var].dims:
            if time is not None:
                cntry_ds = cntry_ds.sel(time=time, method='nearest')
                ds_var = ds[var].sel(time=time, method='nearest').drop_vars("time")
            else:
                latest_time = ds[var]['time'][-1].values
                if verbose:
                    print(f"Time not specified. Selecting the latest available time: {latest_time}")
                cntry_ds = cntry_ds.sel(time=latest_time, method='nearest')
                ds_var = ds[var].sel(time=latest_time, method='nearest').drop_vars("time")
        else:
            cntry_ds = cntry_ds.sel(time='2020-01-01').drop_vars("time")
            ds_var = ds[var]

        # Handle grid_area
        if grid_area:
            if "grid_area" in ds.data_vars:
                grid_ds = ds["grid_area"]
            else:
                if verbose:
                    print("grid_area not found in the dataset. Creating a new one.")
                gdf = create.create_gridded_polygon(resolution=resolution, grid_area="yes")
                grid_ds = gridded_poly_2_dataset(polygon_gdf=gdf, grid_value="grid_area", resolution=resolution)

            ds_var = ds_var * grid_ds

        # Verbose global gridded summary
        if verbose:
            global_methods = {
                "SUM": ds_var.sum().item(),
                "MEAN": ds_var.mean().item(),
                "MAX": ds_var.max().item(),
                "MIN": ds_var.min().item(),
                "STD": ds_var.std().item(),
            }
            if method.upper() in global_methods:
                print(f"Global gridded stats for {var}: {global_methods[method.upper()]:.2f}")

        # Multiply by country fraction
        ds_var = ds_var * cntry_ds

        # Apply aggregation method
        aggregation_methods = {
            "SUM": ds_var.sum(),
            "MEAN": ds_var.mean(),
            "MAX": ds_var.max(),
            "MIN": ds_var.min(),
            "STD": ds_var.std(),
        }
        if method.upper() in aggregation_methods:
            data = aggregation_methods[method.upper()]
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Extract variable names and values
        variable_names = list(data.data_vars.keys())
        values = [data[var].values.item() for var in variable_names]

        # Create a DataFrame from variable names and values
        df = pd.DataFrame({'ISO3': variable_names, var: values})

        # Verbose global tabular summary for cross-checking
        if verbose:
            if method.lower() == 'sum':
                tabular_stat = df[var].sum()
            elif method.lower() == 'mean':
                tabular_stat = df[var].mean()
            elif method.lower() == 'min':
                tabular_stat = df[var].min()   
            elif method.lower() == 'max':
                tabular_stat = df[var].max()       
            elif method.lower() == 'std':
                tabular_stat = df[var].std() 
            else:
                raise ValueError(f"Unsupported fold function: {method}")

        if verbose:
            print(f"Global tabular stats for {var}: {tabular_stat:.2f}")

        dataframes.append(df)

    # Merge all DataFrames horizontally on 'ISO3'
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='ISO3', how='outer')

    # Perform aggregation if specified
    if aggregation:
        continent_df = pd.merge(merged_df, iso3_continent_df[["ISO-alpha3 code", aggregation]],
                                left_on="ISO3", right_on="ISO-alpha3 code")
        merged_df = continent_df.groupby(aggregation).agg({var: method.lower() for var in variables_list}).reset_index()

    return merged_df


def check_iso3_with_country_ds(df, resolution_str):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")
    
    if resolution_str == "1" or resolution_str == "1.0":
        cntry = xr.open_dataset(os.path.join(data_dir, "country_fraction.1deg.2000-2023.a.nc"))   
    elif resolution_str == "0.5":
        cntry = xr.open_dataset(os.path.join(data_dir, "country_fraction.0_5deg.2000-2023.a.nc"))
    elif resolution_str == "0.25":
        cntry = xr.open_dataset(os.path.join(data_dir, "country_fraction.0_25deg.2000-2023.a.nc")) 
    else:
        raise ValueError("Please re-grid the netcdf file to 1, 0.5 or 0.25 degree.")
    
    cntry_vars = [var for var in cntry.variables if var not in cntry.coords]
    df_list = list(df["ISO3"].unique())
    # Find unmatched ISO3 countries
    unmatched_iso3 = list(set(df_list) - set(cntry_vars))
    # Check if the list is not empty before printing
    if unmatched_iso3:
        print(f"Country Not Found: {unmatched_iso3}")



# Define a function to convert ISO3 based on occupation or previous control, given a specific year
def convert_iso3_by_year(df, year):
    # Normalize the year format using pandas' to_datetime and extract the year
    try:
        normalized_year = pd.to_datetime(year, errors='coerce').year  # Extract year from any date format
        if pd.isna(normalized_year):
            raise ValueError(f"Invalid year format: {year}")
    except Exception as e:
        raise ValueError(f"Error in parsing the year: {year}. Error: {e}")
    
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Define a dictionary with ISO3 replacements based on occupation and the year of independence
    replacements = {
        'TLS': {'start_year': 2002, 'previous_iso': 'IDN'},  # Timor-Leste -> Indonesia before 2002
        'XKX': {'start_year': 2008, 'previous_iso': 'SRB'},  # Kosovo -> Serbia before 2008
        'SSD': {'start_year': 2011, 'previous_iso': 'SDN'},  # South Sudan -> Sudan before 2011
    }

    # Iterate over the DataFrame rows
    for index, row in df_copy.iterrows():
        iso3 = row['ISO3']
        
        # Check if the ISO3 is in the replacement dictionary
        if iso3 in replacements:
            # If the normalized year is before the country's independence, replace ISO3 with the occupying country's ISO3
            if normalized_year < replacements[iso3]['start_year']:
                df_copy.loc[index, 'ISO3'] = replacements[iso3]['previous_iso']
    
    # Group by 'ISO3' and sum the values for all numerical columns
    df_copy = df_copy.groupby('ISO3', as_index=False).sum(numeric_only=True)

    return df_copy


def detect_time_step(diffs):
    median_diff = np.median(diffs)
    seconds = median_diff / np.timedelta64(1, 's')
    if np.isclose(seconds, 86400, atol=3600):
        return 'Daily'
    elif np.isclose(seconds, 2628000, atol=86400):
        return 'Monthly'
    elif np.isclose(seconds, 31536000, atol=86400):
        return 'Yearly'
    elif seconds < 3600:
        return 'Hourly or Less'
    else:
        return f'{seconds} seconds'
    
    