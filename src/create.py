import os
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import pyproj
import numpy as np
import xarray as xr
import warnings
import calculate
import utils

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
        if isinstance(geom, Polygon):
            # Calculate the geodesic area for polygons
            area = abs(geod.geometry_area_perimeter(geom)[0]) #  / 1e6  Convert from square meters to square kilometers
            areas.append(area)
            lengths.append(None)  # No length for polygons
        elif isinstance(geom, LineString):
            # Calculate the geodesic length for lines
            length = geod.geometry_length(geom)  #  / 1e3 Convert from meters to kilometers
            lengths.append(length)
            areas.append(None)  # No area for lines

    # Add the new attributes to the GeoDataFrame
    gdf['g_area'] = areas
    gdf['g_length'] = lengths

    # Remove columns with all None values
    if gdf['g_area'].isnull().all():
        gdf.drop(columns=['g_area'], inplace=True)
    if gdf['g_length'].isnull().all():
        gdf.drop(columns=['g_length'], inplace=True)

    return gdf

def create_gridded_polygon(cell_size, out_polygon_path=None, grid_area=False):
    """
    Create a gridded polygon shapefile with the specified cell size.

    Parameters:
    -----------
    cell_size : float
        Size of each grid cell in degrees.

    polygon_path : str, optional
        Path to save the created polygon shapefile. If None, a temporary path is used.

    grid_area : bool, optional
        If True, calculate and add a 'g_area' field to store the geodesic area of each grid cell.

    Returns:
    --------
    world_shp : str
        Path to the generated gridded polygon shapefile.
    """
    # Define the extent of the world
    xmin, ymin, xmax, ymax = -180, -90, 180, 90
    
    # Create grid cells
    cols = np.arange(xmin, xmax, cell_size)
    rows = np.arange(ymin, ymax, cell_size)
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))
    
    # Create a GeoDataFrame
    grid = gpd.GeoDataFrame({'geometry': polygons})
    
    # Set CRS to WGS84
    grid.set_crs(epsg=4326, inplace=True)
    
    # Add 'id' field
    grid['uid'] = range(1, len(grid) + 1)
    
    # Calculate and add 'g_area' field if needed
    if grid_area:
        grid = calculate.calculate_geometry_attributes(input_gdf=grid, column_name="grid_area")
    
    # Save to shapefile
    if out_polygon_path:
        cell_size_str = utils.replace_special_characters(str(cell_size))
        filename = "World_" + cell_size_str + "deg.shp"
        grid.to_file(out_polygon_path + filename)    
    return grid


def create_global_xarray_dataset(pixel_width_deg, pixel_height_deg):
    # Generate latitude and longitude values
    latitudes = np.arange(pixel_height_deg / 2 - 90, 90, pixel_height_deg)[::-1]
    longitudes = np.arange(pixel_width_deg / 2 - 180, 180, pixel_width_deg)

    # Calculate pixel areas using vectorized operations
    lon, lat = np.meshgrid(longitudes, latitudes)
    pixel_areas = np.vectorize(calculate.calculate_geodetic_pixel_area)(
        lon, lat, pixel_width_deg, pixel_height_deg
    )

    ds = xr.Dataset(
        {"grid_area": (['lat', 'lon'], pixel_areas.astype(np.float64))},
        coords={'lat': latitudes, 'lon': longitudes}
    )

    # add attributes
    attrs = {'long_name': "Area of Grids", 'units': "m2"}
    ds["grid_area"].attrs = attrs

    decimal_places = 3  # Adjust as needed
    ds['lon'] = ds['lon'].round(decimal_places)
    ds['lat'] = ds['lat'].round(decimal_places)
    
    return ds

def create_temp_folder(input_path, folder_name="temp"):
    """
    Create a temporary folder in the parent directory of the input path.

    Parameters
    ----------
    input_path : str
        The input path to determine the parent directory.
    folder_name : str, optional
        The name of the temporary folder to be created. Default is "temp".

    Returns
    -------
    str
        The path to the created or existing temporary folder.
    """    
    parent_dir = os.path.dirname(os.path.dirname(input_path))
    path = os.path.join(parent_dir, folder_name)

    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, '')
    return path


def create_new_ds(input_ds, tabular_column, country_ds, netcdf_variable, input_df, verbose):

    # Multiply the country dataset by the input dataset for the specified netCDF variable

    country_netcdf = country_ds * input_ds[netcdf_variable].fillna(0)
    new_ds = xr.Dataset(coords=input_ds.coords)

    # Initialize a DataFrame to store results
    df = pd.DataFrame(columns=["ISO3", "value", "evenly_dis"])

    for var_name in country_netcdf.variables:
        if var_name in input_df["ISO3"].values:
            # Get the corresponding numeric value from the DataFrame
            numeric_value = input_df.loc[input_df["ISO3"] == var_name, tabular_column].values[0]
            total_country = country_netcdf[var_name].sum().item()

            if numeric_value > 0 and total_country == 0:
                # If numeric_value is positive and the total country value is zero
                country_ds_copy = country_ds[var_name].copy()
                netcdf_da = xr.where(country_ds_copy != 0, 1, country_ds_copy)
                new_country_netcdf = country_ds_copy * netcdf_da
                new_country_netcdf = new_country_netcdf.to_dataset()
                total_country = new_country_netcdf[var_name].sum().item()
                
                # Calculate the new dataset value
                new_ds[var_name] = (new_country_netcdf[var_name] * numeric_value) / total_country
                
                # Add to the DataFrame
                total_value = new_ds[var_name].sum().item()
                new_row = pd.DataFrame({"ISO3": [var_name], "value": [total_value], "evenly_dis": [True]})
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                # For cases where a dasymmetric equation is used
                new_ds[var_name] = (country_netcdf[var_name] * numeric_value) / total_country
                
                # Add to the DataFrame
                total_value = new_ds[var_name].sum().item()
                new_row = pd.DataFrame({"ISO3": [var_name], "value": [total_value], "evenly_dis": [False]})
                df = pd.concat([df, new_row], ignore_index=True)

    # Verbose output for evenly distributed countries
    if verbose:
        evenly_df = df[df["evenly_dis"] == True]  # Corrected filtering syntax
        if not evenly_df.empty:  # Check if there are any evenly distributed countries
            print(f"List of evenly distributed countries: {evenly_df['ISO3'].unique()}")
            percentage = (evenly_df["value"].sum() * 100) / df["value"].sum()
            print(f"Evenly distributed country coverage: {percentage:.2f}%")

    return new_ds


