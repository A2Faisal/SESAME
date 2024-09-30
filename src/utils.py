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

import warnings
import create
import calculate
import get

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

warning_message = (
    "The land fraction data is not available at the resolution you requested, which prevents taking into account the portion of coastal grid cells that are not land.\n"
    "Instead, the calculated values will be referenced to the total grid area per cell."
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


def add_variable_attributes(ds, variable_name, long_name, units, source=None, time=None, cell_size=1, value_per_area=False, zero_is_value=False):
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

    if zero_is_value and zero_is_value.upper() == "YES":
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

def gridded_poly_2_dataset(polygon_gdf, grid_value, cell_size, variable_name=None):

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Ensure the coordinate system is WGS84
    polygon_gdf.set_crs(epsg=4326, inplace=True)
    polygon_gdf = polygon_gdf.to_crs(epsg=4326)

    # Extract latitudes and longitudes from the geometry column
    polygon_gdf['lon'] = polygon_gdf['geometry'].centroid.x
    polygon_gdf['lat'] = polygon_gdf['geometry'].centroid.y

    num_lon_points = int(360 / cell_size)
    num_lat_points = int(180 / cell_size)

    lons = np.linspace(-180 + cell_size/2, 180 - cell_size/2, num_lon_points)
    lats = np.linspace(-90 + cell_size/2, 90 - cell_size/2, num_lat_points)

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

def add_grid_variables(ds, cell_size, variable_name, value_per_area):
    
    # Ignore FutureWarning for other functions
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Ensure UserWarning is always shown during this function
    warnings.simplefilter('always', UserWarning)
    
    cell_size_str = str(cell_size)
    if cell_size_str == "1" or cell_size_str == "1.0":
        # Add grid area variable
        base_directory = os.path.dirname(os.path.abspath(__file__))        
        land_frac = xr.load_dataset(os.path.join(base_directory, "G.land_sea_mask.nc"))
        # Merge with the dataset
        ds = xr.merge([ds, land_frac])       
        if value_per_area:
            ds[variable_name] = ds[variable_name] / land_frac["land_area"]
    else:
        gdf = create.create_gridded_polygon(cell_size=cell_size, grid_area="yes")
        grid_ds = gridded_poly_2_dataset(polygon_gdf=gdf, grid_value="grid_area", cell_size=cell_size)
        attrs = {'long_name': "Area of Grids", 'units': "m2"}
        grid_ds["grid_area"].attrs = attrs
        ds = xr.merge([ds, grid_ds])
        if value_per_area:
            # Issue a warning if land fraction data is not available at the desired resolution
            warnings.warn(warning_message, UserWarning)
            ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    
    # Restore the warning filter to its default state
    warnings.simplefilter('default', UserWarning)

    return ds

def gridded_poly_2_xarray(polygon_gdf, grid_value, long_name, units, cell_size, source=None, time=None, variable_name=None, value_per_area=False, zero_is_value=False):

    ds = gridded_poly_2_dataset(polygon_gdf=polygon_gdf, grid_value=grid_value, variable_name=variable_name, cell_size=cell_size)
    variable_name = replace_special_characters(variable_name)

    ds = add_grid_variables(ds=ds, cell_size=cell_size, variable_name=variable_name, value_per_area=value_per_area)
    
    # merge with the dataset
    variable_name = variable_name if variable_name else grid_value
    ds = add_variable_attributes(ds=ds, variable_name=variable_name, long_name=long_name, 
                                     units=units, source=source, time=time, cell_size=cell_size, value_per_area=value_per_area, zero_is_value=zero_is_value)

    return ds

def da_to_ds(da, variable_name, long_name, units, source=None, time=None, cell_size=1, zero_is_value=False, value_per_area=False):
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
    cell_size : float, optional
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
    cell_size = abs(float(ds['lat'].diff('lat').values[0]))
    ds = add_grid_variables(ds=ds, cell_size=cell_size, variable_name=variable_name, value_per_area=value_per_area)

    # Add variable metadata
    attrs = {'long_name': long_name, 'units': units}
    if source is not None:
        attrs['source'] = source
        
    ds[variable_name].attrs = attrs
    # Set and add global attributes
    ds.attrs = global_attr
    
    return ds

def determine_long_name_point(fold_field, variable_name, long_name, fold_function):
    if long_name is None:
        if fold_field is None or (fold_function is not None and fold_function.lower() == 'sum'):
            return reverse_replace_special_characters(variable_name) if variable_name else "count"
    return long_name if long_name else "count"


def determine_units_line(units, value_per_area):
    if units == "meter/grid-cell" and value_per_area:
        return "m m-2"
    return units

def determine_long_name_line(fold_field, variable_name, long_name, fold_function):
    if long_name is None:
        if fold_field is None or (fold_function is not None and fold_function.lower() == 'sum'):
            return reverse_replace_special_characters(variable_name) if variable_name else reverse_replace_special_characters(f"length_{fold_function.lower()}")
    return long_name if long_name else reverse_replace_special_characters(f"length_{fold_function.lower()}")

def determine_long_name_line(long_name, fold_field, variable_name):
    if long_name is None:
        if variable_name:
            long_name = reverse_replace_special_characters(variable_name)
        else:
            long_name = reverse_replace_special_characters(fold_field)
    return long_name
        

def dataframe_stats_line(dataframe, fold_field=None, fold_function="sum", verbose=False):
    if fold_function.lower() == "sum":
        if fold_field is None or fold_field == "length_m":
            dataframe = calculate.calculate_geometry_attributes(dataframe)
            global_summary_stats = dataframe['length_m'].sum()
            if verbose:
                print(f"Global stats before gridding: {global_summary_stats:.2f}")
            return global_summary_stats * 1e-3
    else:
        raise ValueError(f"Unsupported fold_function: {fold_function}. Choose 'sum'. or set verbose=False")
    
def determine_units_poly(units, value_per_area, fraction):
    if fraction:
        if fraction and not value_per_area:
            return "fraction"
        if fraction and value_per_area:
            raise ValueError("Fraction and value per area cannot be created together.")
    elif value_per_area:
        units = "m2 m-2"
    return units
    

def determine_long_name_poly(variable_name, long_name, fold_function):
    if long_name is None:
        if variable_name is None or (fold_function is not None and fold_function.lower() == 'sum'):
            return reverse_replace_special_characters(variable_name) if variable_name else "area"
    return long_name if long_name else "area"


def dataframe_stats_poly(dataframe, fold_function="sum"):
    if fold_function.lower() == "sum":
        dataframe = calculate.calculate_geometry_attributes(dataframe)
        global_summary_stats = dataframe['area_m2'].sum()
        return global_summary_stats * 1e-6
    else:
        raise ValueError(f"Unsupported fold_function: {fold_function}. Choose 'sum'. or set verbose=False")
    

def determine_units_point(units, value_per_area):
    if units == "value/grid-cell" and value_per_area:
        return "value m-2"
    return units


def dataframe_stats_point(dataframe, fold_field=None, fold_function="sum"):
    if fold_field is None or fold_field == "count":
        if fold_function is None or fold_function.lower() == 'sum':
            global_summary_stats = len(dataframe)
        else:
            raise ValueError(f"Unsupported fold_function: {fold_function}")
    elif fold_function is not None and fold_function.lower() == 'sum' and fold_field is not None:
        global_summary_stats = dataframe[fold_field].sum()
    else:
        raise ValueError(f"Unsupported combination of fold_field: {fold_field} and fold_function: {fold_function}")
    return global_summary_stats


def xarray_dataset_stats(dataset, variable_name=None, fold_field=None, value_per_area=None, cell_size=1):
    if variable_name is None and fold_field:
        variable_name = fold_field
    elif variable_name and fold_field:
        variable_name = variable_name
    if value_per_area:
        cell_size_str = str(cell_size)
        if cell_size_str == "1" or cell_size_str == "1.0":
            global_gridded_stats = (dataset[variable_name].fillna(0) * dataset["land_area"]).sum().item()
        else:
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
            
def point_spatial_join(polygons_gdf, points_gdf, fold_field=None, fold_function='sum', x_offset=0.0001, y_offset=0.0001):
    # Adjust points to ensure they are within or correctly intersecting polygons
    points_gdf = adjust_points(points_gdf, polygons_gdf, x_offset, y_offset)
    
    # Perform spatial join with 'intersects' operation
    points_within_polygons = gpd.sjoin(points_gdf, polygons_gdf, predicate='intersects')
    
    # If field_name is provided, convert the column to numeric
    if fold_field:
        points_within_polygons[fold_field] = pd.to_numeric(points_within_polygons[fold_field], errors='coerce')
        variable_name = replace_special_characters(fold_field)
        # Group by polygon and compute summary statistic based on operation
        if fold_function.lower() == 'sum':
            summary_stats = points_within_polygons.groupby('index_right')[fold_field].sum().reset_index(name=variable_name)
        elif fold_function.lower() == 'mean':
            summary_stats = points_within_polygons.groupby('index_right')[fold_field].mean().reset_index(name=variable_name)
        elif fold_function.lower() == 'max':
            summary_stats = points_within_polygons.groupby('index_right')[fold_field].max().reset_index(name=variable_name)
        elif fold_function.lower() == 'min':
            summary_stats = points_within_polygons.groupby('index_right')[fold_field].min().reset_index(name=variable_name)            
        elif fold_function.lower() == 'std':
            summary_stats = points_within_polygons.groupby('index_right')[fold_field].std().reset_index(name=variable_name)
        else:
            raise ValueError(f"Unsupported operation: {fold_field}. Choose from 'sum', 'mean', 'max', 'std'.")
    
        # Merge summary statistics with polygons GeoDataFrame
        polygons_gdf = polygons_gdf.merge(summary_stats, how='left', left_index=True, right_on='index_right')
        print(polygons_gdf[variable_name].sum())
    
    else:
        # Count the points within each polygon
        polygon_counts = points_within_polygons.groupby('index_right').size().reset_index(name='count')
        # Add the count to the polygons GeoDataFrame
        polygons_gdf['count'] = polygons_gdf.index.to_series().map(polygon_counts.set_index('index_right')['count']).fillna(0).astype(int)
    
    return polygons_gdf

def point_spatial_join(polygons_gdf, points_gdf, fold_field=None, fold_function='sum', x_offset=0.0001, y_offset=0.0001):
    # Ensure both GeoDataFrames use the same CRS
    if polygons_gdf.crs != points_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
        
    # Adjust points to ensure they are within or correctly intersecting polygons
    points_gdf = adjust_points(points_gdf, polygons_gdf, x_offset, y_offset)
    
    # Perform intersection
    intersections = gpd.overlay(points_gdf, polygons_gdf, how='intersection')
    
    # If fold_field is provided, convert the column to numeric
    if fold_field:
        intersections[fold_field] = pd.to_numeric(intersections[fold_field], errors='coerce')
        # Group by polygon and compute summary statistic based on operation
        if fold_function.lower() == 'sum':
            intersections = intersections.groupby('uid')[fold_field].sum().reset_index()
        elif fold_function.lower() == 'mean':
            intersections = intersections.groupby('uid')[fold_field].mean().reset_index()
        elif fold_function.lower() == 'max':
            intersections = intersections.groupby('uid')[fold_field].max().reset_index()
        elif fold_function.lower() == 'min':
            intersections = intersections.groupby('uid')[fold_field].min().reset_index()
        elif fold_function.lower() == 'std':
            intersections = intersections.groupby('uid')[fold_field].std().reset_index()
        else:
            raise ValueError(f"Unsupported operation: {fold_field}. Choose from 'sum', 'mean', 'max', min, 'std'.")

    else:
        intersections = intersections.groupby('uid').size().reset_index(name='count')
    
    joined_gdf = polygons_gdf.merge(intersections, on='uid', how='left')
    return joined_gdf
    

def line_intersect(polygons_gdf, lines_gdf, fold_field=None, fold_function='sum'):
    
    # Ensure both GeoDataFrames use the same CRS
    if polygons_gdf.crs != lines_gdf.crs:
        lines_gdf = lines_gdf.to_crs(polygons_gdf.crs)
    
    # Calculate geometry attributes
    polygons_gdf = calculate.calculate_geometry_attributes(input_gdf=polygons_gdf, column_name="grid_area")
    
    # Perform intersection
    intersections = gpd.overlay(lines_gdf, polygons_gdf, how='intersection')
    
    # Calculate geometry attributes for intersections
    intersections = calculate.calculate_geometry_attributes(input_gdf=intersections, column_name="length_m")

    # If fold_field is provided, convert the column to numeric
    if fold_field:
        intersections[fold_field] = pd.to_numeric(intersections[fold_field], errors='coerce')
        # variable_name = replace_special_characters(fold_field)
        # Group by polygon and compute summary statistic based on operation
        if fold_function.lower() == 'sum':
            intersections = intersections.groupby('uid')[fold_field].sum().reset_index()
        elif fold_function.lower() == 'mean':
            intersections = intersections.groupby('uid')[fold_field].mean().reset_index()
        elif fold_function.lower() == 'max':
            intersections = intersections.groupby('uid')[fold_field].max().reset_index()
        elif fold_function.lower() == 'min':
            intersections = intersections.groupby('uid')[fold_field].min().reset_index()
        elif fold_function.lower() == 'std':
            intersections = intersections.groupby('uid')[fold_field].std().reset_index()
        else:
            raise ValueError(f"Unsupported operation: {fold_field}. Choose from 'sum', 'mean', 'max', min, 'std'.")

    else:
        if fold_function.lower() == "sum":
            intersections = intersections.groupby('uid')['length_m'].sum().reset_index()
        elif fold_function.lower() == "mean":
            intersections = intersections.groupby('uid')['length_m'].mean().reset_index()
        elif fold_function.lower() == "max":
            intersections = intersections.groupby('uid')['length_m'].max().reset_index()
        elif fold_function.lower() == "min":
            intersections = intersections.groupby('uid')['length_m'].min().reset_index()
        elif fold_function.lower() == "std":
            intersections = intersections.groupby('uid')['length_m'].std().reset_index()
    
    joined_gdf = polygons_gdf.merge(intersections, on='uid', how='left')
    return joined_gdf


def convert_xarray_to_gdf(ds, variable_name, cell_size=1):
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
                (lons[j] - cell_size / 2, lats[i] - cell_size / 2),  # Bottom left corner
                (lons[j] + cell_size / 2, lats[i] - cell_size / 2),  # Bottom right corner
                (lons[j] + cell_size / 2, lats[i] + cell_size / 2),  # Top right corner
                (lons[j] - cell_size / 2, lats[i] + cell_size / 2)   # Top left corner
            ])
            polygons.append(polygon)
            values.append(data[i, j])  # Use the corresponding value
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'land_area': values  # Add the values as a column
    }, crs='EPSG:4326')  # Set the coordinate reference system

    return gdf


def poly_fraction(ds, variable_name, cell_size, polygons_gdf=None):

    # Ignore FutureWarning for other functions
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Ensure UserWarning is always shown during this function
    warnings.simplefilter('always', UserWarning)

    # Save the attributes of the variable
    attrs = ds[variable_name].attrs

    cell_size_str = str(cell_size)
    if cell_size_str == "1" or cell_size_str == "1.0":
        base_directory = os.path.dirname(os.path.abspath(__file__))        
        land_frac = xr.load_dataset(os.path.join(base_directory, "G.land_sea_mask.nc"))
        ds = xr.merge([ds, land_frac])
        # ensure there is no grid values if land fraction is 0
        land_frac_da = xr.where(ds["land_fraction"] > 0, 1, ds["land_fraction"])
        ds[variable_name] = ds[variable_name] * land_frac_da
        # Compute the new fraction using the maximum of ds[variable_name] and ds["land_area"]
        ds[variable_name] = ds[variable_name] / np.maximum(ds[variable_name], ds["land_area"])
        # Ensure no values are greater than 1, keeping NaNs unchanged
        ds[variable_name] = ds[variable_name].where(ds[variable_name].isnull() | (ds[variable_name] <= 1), 1)

    else:
        # Issue a warning if land fraction data is not available at the desired resolution
        warnings.warn(warning_message, UserWarning)
        grid_ds = gridded_poly_2_dataset(polygon_gdf=polygons_gdf, grid_value="grid_area", cell_size=cell_size)
        attrs = {'long_name': "Area of Grids", 'units': "m2"}
        grid_ds["grid_area"].attrs = attrs
        ds = xr.merge([ds, grid_ds])
        ds[variable_name] = ds[variable_name] / grid_ds["grid_area"]
    
    # Reassign the saved attributes back to the variable
    ds[variable_name].attrs = attrs
    
    # Restore the warning filter to its default state
    warnings.simplefilter('default', UserWarning)
    return ds

def poly_intersect(poly_gdf, polygons_gdf, variable_name, long_name, units, source, time, cell_size, fold_function="sum", value_per_area=None, zero_is_value=None, fraction=False):
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Calculate geometry attributes
    poly_gdf = calculate.calculate_geometry_attributes(input_gdf=poly_gdf, column_name="raw_area")
    polygons_gdf = calculate.calculate_geometry_attributes(input_gdf=polygons_gdf, column_name="grid_area")
    # Perform intersection
    intersections = gpd.overlay(poly_gdf, polygons_gdf, how='intersection')
    # Calculate geometry attributes for intersections
    intersections = calculate.calculate_geometry_attributes(input_gdf=intersections, column_name="in_area")

    if fold_function.lower() == "sum":
        intersections = intersections.groupby('uid')['in_area'].sum().reset_index()
    elif fold_function.lower() == "mean":
        intersections = intersections.groupby('uid')['in_area'].mean().reset_index()
    elif fold_function.lower() == "max":
        intersections = intersections.groupby('uid')['in_area'].max().reset_index()
    elif fold_function.lower() == "min":
        intersections = intersections.groupby('uid')['in_area'].min().reset_index()
    elif fold_function.lower() == "std":
        intersections = intersections.groupby('uid')['in_area'].std().reset_index()

    joined_gdf = polygons_gdf.merge(intersections, on='uid', how='left')

    ds = gridded_poly_2_xarray(
                polygon_gdf=joined_gdf,
                grid_value='in_area',
                long_name=long_name,
                units=units,
                source=source,
                time=time,
                cell_size=cell_size,
                variable_name=variable_name,
                value_per_area=value_per_area,
                zero_is_value=zero_is_value
            )

    if fraction:
        ds = poly_fraction(ds=ds, variable_name=variable_name, cell_size=cell_size, polygons_gdf=polygons_gdf)

    return ds



def netcdf_2_tif(netcdf_path, netcdf_variable, time=None):
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
    # create a temp path
    temp_path = create.create_temp_folder(netcdf_path, folder_name="temp")
    
    # Open the NetCDF file with xarray
    ds = xr.open_dataset(netcdf_path)
    
    # Select the variable
    data = ds[netcdf_variable]
    
    # Handle multidimensional data and specific time
    if time is not None:
        data = data.sel(time=time, method="nearest").drop_vars("time")
    
    # Extract the data array
    array = data.values
    
    # identify lat, lon variables
    x_dimension, y_dimension = get.identify_lat_lon_names(netcdf_path)
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
    
    transform = from_origin(min_lon, max_lat, (max_lon - min_lon) / width, (max_lat - min_lat) / height)

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
    raster_layer = f"{netcdf_variable}_{time[:10] if time else ''}.tif"
    output_raster = os.path.join(temp_path, raster_layer)
    
    # Write the data to a TIFF file
    with rasterio.open(output_raster, 'w', **metadata) as dst:
        dst.write(array, 1)
    
    return output_raster



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
            src_res = (src_transform[0], src_transform[4])  # (pixel_width, pixel_height)

            # Calculate the dimensions of the output raster based on the extent and input resolution
            dst_width = int((dst_extent[2] - dst_extent[0]) / src_res[0])
            dst_height = int((dst_extent[3] - dst_extent[1]) / abs(src_res[1]))

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
            src_res = (transform[0], transform[4])
            dst_width = int((dst_extent[2] - dst_extent[0]) / src_res[0])
            dst_height = int((dst_extent[3] - dst_extent[1]) / abs(src_res[1]))
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
    x_cell_size = (dst_extent[2] - dst_extent[0]) / dst_width
    y_cell_size = (dst_extent[3] - dst_extent[1]) / dst_height

    return dst_array, x_cell_size, y_cell_size



def regrid_array_2_ds(array, fold_function, variable_name, long_name, units="value/grid-cell", source=None, cell_size=1,
                     time=None, zero_is_value=None, value_per_area=False, verbose=False):
    
    arr = array.astype(np.float64)

    # Calculate raw global value based on the conversion type
    if fold_function.upper() == 'SUM':
        raw_global_value = arr.sum()

    # Get dimensions and calculate grid resolution
    num_rows, num_cols = arr.shape
    num_lat, num_lon = calculate.calculate_grid_resolution(resolution=cell_size)
    
    # Set a tolerance level
    tolerance = 0.01
    
    # Check if the dimensions are perfectly divisible within the tolerance
    if abs(num_rows / num_lat - round(num_rows / num_lat)) < tolerance and abs(num_cols / num_lon - round(num_cols / num_lon)) < tolerance:
        padded_arr = arr
    else:
        # Calculate padding needed for the array
        lat_padding = num_lat - (arr.shape[0] % num_lat)
        lon_padding = num_lon - (arr.shape[1] % num_lon)
    
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

    lat_resolution = cell_size
    lon_resolution = cell_size
    lat = np.linspace(90 - lat_resolution / 2, -90 + lat_resolution / 2, num=num_lat)
    lon = np.linspace(-180 + lon_resolution / 2, 180 - lon_resolution / 2, num=num_lon)


    # Create an xarray DataArray with dimensions and coordinates
    da = xr.DataArray(aligned_arr, dims=("lat", "lon", "lat_factor", "lon_factor"),
                      coords={"lat": lat, "lon": lon})

    # Perform the aggregation over the lat_factor and lon_factor dimensions
    if fold_function.upper() == 'SUM':
        da_agg = da.sum(dim=['lat_factor', 'lon_factor'])
    elif fold_function.upper() == 'MEAN':
        if zero_is_value and zero_is_value.upper() == "YES":
            da_agg = da.mean(dim=['lat_factor', 'lon_factor'])
        else:
            # Calculate the sum and count of non-zero values
            da_sum = da.sum(dim=['lat_factor', 'lon_factor'])
            da_count = da.where(da != 0).count(dim=['lat_factor', 'lon_factor'])
            # Calculate the mean, handling division by zero
            da_agg_nan = da_sum / da_count.where(da_count != 0, np.nan)
            # Replace nan values with 0
            da_agg = da_agg_nan.fillna(0)
    elif fold_function.upper() == 'MAX':
        da_agg = da.max(dim=['lat_factor', 'lon_factor'])
    elif fold_function.upper() == 'MIN':
        da_agg = da.min(dim=['lat_factor', 'lon_factor'])
    elif fold_function.upper() == 'STD':
        da_agg = da.std(dim=['lat_factor', 'lon_factor'])
    else:
        raise ValueError("Conversion should be either SUM, MEAN, MAX, MIN or STD")

    if verbose and fold_function.upper() == 'SUM':
        print(f"Raw global {fold_function}: {raw_global_value:.3f}")
        regridded_global_value = da_agg.sum().item()
        print(f"Re-gridded global {fold_function}: {regridded_global_value:.3f}")

    da = da_agg
    
    # convert dataarray to dataset
    ds = da_to_ds(da, variable_name, long_name, units, source, time, cell_size, zero_is_value, value_per_area)

    return ds



def xy_not_eq(raster_path, fold_function, variable_name, long_name, units, source=None, time=None, cell_size=1, 
                                 value_per_area=False, zero_is_value=False):
    # Convert raster to polygon GeoDataFrame
    gdf = raster_to_polygon_gdf(raster_file=raster_path)
    gdf = gdf.fillna(0)
    gdf = calculate.calculate_geometry_attributes(input_gdf=gdf, column_name="ras_area")
    # Create gridded polygon GeoDataFrame
    polygons_gdf = create.create_gridded_polygon(cell_size=cell_size)
    # Perform intersection
    intersections = gpd.overlay(gdf, polygons_gdf, how='intersection')
    # Calculate geometry attributes
    intersections = calculate.calculate_geometry_attributes(input_gdf=intersections, column_name="in_area")
    # Calculate the fraction
    intersections["frac"] = intersections["in_area"] / intersections["ras_area"]
    # compute statistics
    result_df = compute_weighted_statistics(gdf=intersections, stat=fold_function)
    # Merge results with polygons_gdf
    joined_gdf = polygons_gdf.merge(result_df, on='uid', how='left')
    variable_name = replace_special_characters(variable_name)
    ds = gridded_poly_2_xarray(polygon_gdf=joined_gdf, grid_value=fold_function, long_name=long_name, 
                                 units=units, source=source, time=time, variable_name=variable_name, cell_size=cell_size, 
                                 value_per_area=value_per_area, zero_is_value=zero_is_value)
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



def tif_2_ds(input_raster, variable_name, fold_function, long_name, units="value/grid-cell", source=None, cell_size=1,
                     time=None, zero_is_value=None, value_per_area=False, resampling_method='bilinear', verbose=False):
    
    # Step-1: Check the cell size
    # Open the input raster using rasterio
    with rasterio.open(input_raster) as raster:
        # Get the X and Y cell sizes from the raster properties
        x_size, y_size = raster.res[0], raster.res[1]

        # Round and convert cell sizes to float
        x_size = round(float(x_size), 3)
        y_size = round(float(y_size), 3)

        if x_size != y_size or cell_size > x_size:
            ds = xy_not_eq(raster_path=input_raster, variable_name=variable_name, fold_function=fold_function, 
                           long_name=long_name, units=units, source=source, time=time, cell_size=cell_size, 
                           value_per_area=value_per_area, zero_is_value=zero_is_value)
        else:
            # re-grid
            array, x_cell_size, y_cell_size = reproject_and_fill(input_raster)
            ds = regrid_array_2_ds(array=array, fold_function=fold_function, variable_name=variable_name, 
                                   long_name=long_name, units=units, source=source, cell_size=cell_size, 
                                   time=time, zero_is_value=zero_is_value, value_per_area=value_per_area, 
                                   verbose=verbose)
    
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
        print(f"Successfully deleted the folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting the folder: {e}")

def grid_2_table(input_netcdf_path=None, ds=None, variable=None, time=None, grid_area=None, cell_size=1, aggregation=None, method='sum', verbose=False):

    base_directory = os.path.dirname(os.path.abspath(__file__))

    if input_netcdf_path:
        ds = xr.load_dataset(input_netcdf_path)

    if not isinstance(ds, xr.Dataset):
        raise ValueError("Please provide either netcdf or xarray dataset.")
    
    # Check if a specific variable is provided, otherwise consider all variables in the dataset
    if variable is not None:
        variables_list = [variable]
    else:
        # Get the list of variables except the specified ones
        exclude_vars = ["time", "lat", "lon", "grid_area", "grid_area_1deg"]
        variables_list = [var for var in ds.variables if var not in exclude_vars]
        print(f"List of variables in the dataset: {variables_list}")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Load ISO3 to continent mapping from CSV
    try:
        iso3_continent_df = pd.read_csv(os.path.join(base_directory, "un_geoscheme.csv"), encoding='utf-8')
    except UnicodeDecodeError:
        # Try a different encoding if 'utf-8' fails
        iso3_continent_df = pd.read_csv(os.path.join(base_directory, "un_geoscheme.csv"), encoding='latin1')

    # Loop through each variable in the dataset
    for var in variables_list:   
        # Select the country fraction data based on cell size
        try:
            cell_size_str = str(cell_size)
            if cell_size_str == "1" or cell_size_str == "1.0":
                cntry_ds = xr.load_dataset(os.path.join(base_directory, "country_fraction.1deg.2000-2023.a.nc"))
            elif cell_size_str == "0.5":
                cntry_ds = xr.load_dataset(os.path.join(base_directory, "Country_Fraction.0_5deg.2000-2023.nc"))
        except FileNotFoundError as e:
            print(f" Error while reading file {e} ")
        
        # Check if 'time' is a dimension in the variable
        if 'time' in ds[var].dims:
            if time is not None:
                cntry_ds = cntry_ds.sel(time=time, method='nearest')
                ds_var = ds[var].sel(time=time, method='nearest')
            else:
                latest_time = ds[var]['time'][-1].values
                cntry_ds = cntry_ds.sel(time=latest_time, method='nearest')
                ds_var = ds[var].sel(time=latest_time, method='nearest')
        else:
            cntry_ds = cntry_ds.sel(time='2020-01-01')
            ds_var = ds[var]

        if grid_area is not None and grid_area.upper() == 'YES':
            gdf = create.create_gridded_polygon(cell_size=1, grid_area="yes")
            grid_ds = gridded_poly_2_dataset(polygon_gdf=gdf, grid_value="grid_area", cell_size=cell_size)
            # Multiply the variable by grid area if specified
            ds_var = ds_var * grid_ds["grid_area"]
            print(f"Generating the tabular data for: {var}")
        
        if verbose:
            if method.upper() =="SUM":
                print(f"Global gridded {method}: {ds_var.sum().item()}")
            elif method.upper() =="MEAN":
                print(f"Global gridded {method}: {ds_var.mean().item()}")
            elif method.upper() == "MAX":
                print(f"Global gridded {method}: {ds_var.max().item()}")
            elif method.upper() == "MIN":
                print(f"Global gridded {method}: {ds_var.min().item()}")
            elif method.upper() == "STD":
                print(f"Global gridded {method}: {ds_var.std().item()}")
            else:
                sys.exit(1)
        
        ds_var = ds_var * cntry_ds
#         ds_var = ds_var.drop_vars('time')
        
        if method.upper() =="SUM":
            data = ds_var.sum()
        elif method.upper() =="MEAN":
            data = ds_var.mean()
        elif method.upper() == "MAX":
            data = ds_var.max()
        elif method.upper() == "MIN":
            data = ds_var.min()
        elif method.upper() == "STD":
            data = ds_var.std()
        else:
            sys.exit(1)

        # Extract variable names and values
        variable_names = list(data.data_vars.keys())
        values = [data[var].values.item() for var in variable_names]

        # Create a DataFrame from variable names and values
        df = pd.DataFrame({'ISO3': variable_names, var: values})
        dataframes.append(df)

    # If a specific time is provided, add a 'Year' column to the resulting DataFrame
    if time is not None:
        year = time[:4]
        merged_df = dataframes[0]  # Start with the first DataFrame
        merged_df = pd.concat([merged_df['ISO3'],
                               pd.DataFrame({'Year': year}, index=merged_df.index),
                               merged_df.drop(columns=['ISO3'])], axis=1)
    else:
        merged_df = dataframes[0]  # Start with the first DataFrame
    
    method_upper = method.upper()
    # Merge DataFrames based on 'ISO3' column
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='ISO3')
        
    if aggregation == 'region':
        continent_df = pd.merge(merged_df, iso3_continent_df[['ISO-alpha3 code', 'Region 1']], left_on='ISO3', right_on='ISO-alpha3 code')
        if method_upper in ["SUM", "MEAN", "MAX", "STD"]:
            agg_func = method_upper.lower()  # Method names in DataFrame are lowercase
            merged_df = continent_df.groupby('Region 1').agg({var: agg_func}).reset_index()
        else:
            sys.exit(1)
    if verbose:
        print(f"Global tabular {method_upper}: {getattr(merged_df[var], method)()}")

    return merged_df


def check_iso3_with_country_ds(df, cell_size_str):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    if cell_size_str == "1" or cell_size_str == "1.0":
        cntry = xr.load_dataset(os.path.join(base_directory, "country_fraction.1deg.2000-2023.a.nc"))   
    elif cell_size_str == "0.5":
        cntry = xr.load_dataset(os.path.join(base_directory, "country_fraction.0_5deg.2000-2023.a.nc")) 
    else:
        raise ValueError("Please re-grid the netcdf file to 1 or 0.5 degree.")
    cntry_vars = [var for var in cntry.variables if var not in cntry.coords]
    df_list = list(df["ISO3"].unique())
    # Find unmatched ISO3 countries
    unmatched_iso3 = list(set(df_list) - set(cntry_vars))
    # Check if the list is not empty before printing
    if unmatched_iso3:
        print(f"Country Not Found: {unmatched_iso3}")