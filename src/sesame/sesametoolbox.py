"""
SESAME is an open-source Python tool designed to make spatial data analysis, visualization, and exploration accessible to all.  
Whether you’re a researcher, student, or enthusiast, SESAME helps you unlock insights from geospatial data with just a few lines of code.

---

**What can you do with the SESAME toolbox?**

- Conveniently process and analyze both spatial datasets (e.g. GeoTIFFs) and tabular jurisdictional data (e.g. csv files by country) through a unified set of tools.
- Generate standardized netcdf files from a wide range of spatial input types (e.g. lines, points, polygons)
- Create publication-ready maps and plots.
- Explore spatial and temporal patterns among hundreds of variables in the Human-Earth Atlas.

**Getting Started with the Human-Earth Atlas:**

1. Install SESAME*
2. Download the Human-Earth Atlas ([Figshare Link](https://doi.org/10.6084/m9.figshare.28432499))
3. Load your spatial data (e.g., land cover, population, climate)
4. Use SESAME’s plotting tools to visualize and compare datasets
5. Explore spatial and temporal patterns among hundreds of variables in the Human-Earth Atlas.

*Note: SESAME may take up to 2 minutes to load when used for the first time. This will not recur with further use.

**Navigating the Atlas:**
1. List the netCDF files in the Human–Earth Atlas
```python
import sesame as ssm
ssm.atlas(directory=atlas)
```
<img src="../images/atlas.png" alt="Human-Earth Atlas" width="600"/>

2. View dataset metadata
```python
ssm.list_variables("atlas/B.land.cover.2001-2023.a.nc")
```
<img src="../images/info.png" alt="NetCDF Info" width="600"/>

3. Visualize data on the map
```python
# Load data
netcdf_file = "atlas/T.transportation.roads.nc"
ssm.plot_map(dataset=netcdf_file,variable="roads_gross", color='magma_r', title='Gross Road Mass', label='g m-2', vmin=0, vmax=1e4, extend_max=True)
```
<img src="../images/gross_road.png" alt="Gross Road Mass Map" width="600"/>

4. Quick mathematical operation
```python
# Load data
netcdf_file = "atlas/T.transportation.roads.nc"
# Perform the operation
ssm.divide_variables(dataset=netcdf_file, variable1="road_length", variable2="grid_area", new_variable_name="road_density")
```

Ready to get started? Dive into the function docs below or read [The SESAME Human-Earth Atlas](https://www.nature.com/articles/s41597-025-05087-5) paper for inspiration!

---
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import json

from . import create
from . import utils
from . import calculate
from . import plot
from . import get

# import create
# import utils
# import calculate
# import plot
# import get


def point_2_grid(point_data, variable_name='variable', long_name='variable', units="value/grid-cell", source=None, time=None, resolution=1, agg_column=None, agg_function="sum", attr_field=None, output_directory=None, output_filename=None, normalize_by_area=False, zero_is_value=False, verbose=False):
    
    """
    Converts point data from a shapefile or GeoDataFrame into a gridded netCDF dataset.

    Parameters
    ----------
    - point_data : GeoDataFrame or str. Input point data to be gridded. Can be either a GeoDataFrame or a path to a point shapefile (.shp).
    - variable_name : str, optional. Name of the variable to include in the netCDF attributes metadata. Defaults to:
        - The unique entries in the `attr_field` column if specified.
        - The input filename without extension if `attr_field` and `variable_name` are not specified.
    - long_name : str, optional. A descriptive name for the variable, added to the netCDF metadata. Behaves the same as `variable_name` if
        `attr_field` is specified. Defaults to the input filename without extension if unspecified.
    - units : str, optional. Units of the data variable to include in the netCDF metadata. Default is "value/grid-cell".
    - source : str, optional. String describing the original source of the input data. This will be added to the netCDF metadata.
    - time : str, optional. Time dimension for the output netCDF. If specified, the output will include a time dimension with the
        value provided. Default is None (spatial, 2D netCDF output).
    - resolution : float, optional. Desired resolution for the grid cells in the output dataset. Default is 1 degree.
    - agg_column : str, optional. Column name in the shapefile or GeoDataFrame specifying the values to aggregate in each grid cell.
        Defaults to counting the number of points per grid cell.
    - agg_function : str, optional. Aggregation method for combining values in each grid cell. Options include:
        - 'sum' (default): Sums all point values.
        - 'max': Takes the maximum value.
        - 'min': Takes the minimum value.
        - 'std': Computes the standard deviation.
    - attr_field : str, optional. Column name in the shapefile or GeoDataFrame specifying the variable names for multiple data types.
    - output_directory : str, optional. Directory where the output NetCDF file will be saved. If None, but output_filename is True, the file will be saved in the current working directory.
    - output_filename : str, optional. Name of the output NetCDF file (without the `.nc` extension). If not provided:
        - Uses the input shapefile name if a shapefile path is given.
        - Saves as `"gridded_points.nc"` if a GeoDataFrame is provided as input.
    - normalize_by_area : bool, optional. If True, normalizes the grid values by area (e.g., converts to value per square meter). Default is False.
    - zero_is_value : bool, optional. If True, treats zero values as valid data rather than as no-data. Default is False.
    - verbose : bool, optional. If True, prints information about the process, such as global sum of values before and after gridding. Default is False.

    Returns
    -------
    - xarray.Dataset. Transformed dataset with gridded data derived from the input point data.

    Notes
    -----
    - The function supports input in the form of a shapefile or GeoDataFrame containing point data.
    - If points lie exactly on a grid boundary, they are shifted by 0.0001 degrees in both latitude and longitude to ensure assignment to a grid cell.
    - The function creates a netCDF file, where data variables are aggregated based on the `agg_column` and `agg_function`.
    
    Example
    -------
    >>> point_2_grid(point_data=shapefile_path, 
    ...             variable_name="airplanes", 
    ...             long_name="Airplanes Count", 
    ...             units="airport/grid-cell", 
    ...             source="CIA", 
    ...             resolution=1,
    ...             verbose=True
    ... )
    
    """

    # Determine if input is a path (string or Path) or a GeoDataFrame
    if isinstance(point_data, (str, bytes, os.PathLike)):
        if verbose:
            print("Reading shapefile from path...")
        points_gdf = gpd.read_file(point_data)
    elif isinstance(point_data, gpd.GeoDataFrame):
        points_gdf = point_data
    else:
        raise TypeError("Input must be a GeoDataFrame or a shapefile path (string or Path).")

    # create gridded polygon
    polygons_gdf = create.create_gridded_polygon(resolution=resolution, out_polygon_path=None, grid_area=False)
    
    if attr_field is not None:
        unique_rows = points_gdf[attr_field].unique().tolist()
        dataset_list = []
        
        for filter_var in unique_rows:
            # Filter the GeoDataFrame
            filtered_gdf = points_gdf[points_gdf[attr_field] == filter_var].copy()
            joined_gdf = utils.point_spatial_join(polygons_gdf, filtered_gdf, agg_column=agg_column, agg_function=agg_function)

            # Determine agg_column, long_name, and units for the current iteration
            current_agg_column = agg_column or "count"
            current_long_name = utils.reverse_replace_special_characters(filter_var)
            current_units = utils.determine_units_point(units, normalize_by_area)

            # Convert joined GeoDataFrame to xarray dataset
            ds_var = utils.gridded_poly_2_xarray(
                polygon_gdf=joined_gdf,
                grid_value=current_agg_column,
                long_name=current_long_name,
                units=current_units,
                source=source,
                time=time,
                resolution=resolution,
                variable_name=filter_var,
                normalize_by_area=normalize_by_area,
                zero_is_value=zero_is_value
            )

            # Print or process verbose information
            if verbose:
                global_summary_stats = utils.dataframe_stats_point(dataframe=filtered_gdf, agg_column=current_agg_column, agg_function=agg_function)
                print(f"Global stats of {filter_var} before gridding : {global_summary_stats:.2f}")
                var_name = utils.replace_special_characters(filter_var)
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds_var, variable_name=var_name, normalize_by_area=normalize_by_area, resolution=resolution)
                print(f"Global stats of {filter_var} after gridding: {global_gridded_stats:.2f}")

            print("\n")
            dataset_list.append(ds_var)
        
        # Merge all datasets from different filtered GeoDataFrames
        ds = xr.merge(dataset_list)
        
    else:
        joined_gdf = utils.point_spatial_join(polygons_gdf, points_gdf, agg_column=agg_column, agg_function=agg_function)

        # Determine agg_column, long_name, and units
        agg_column = agg_column or "count"
        long_name = utils.determine_long_name_point(agg_column, variable_name, long_name, agg_function)
        units = utils.determine_units_point(units, normalize_by_area)
        
        ds = utils.gridded_poly_2_xarray(
            polygon_gdf=joined_gdf,
            grid_value=agg_column,
            long_name=long_name,
            units=units,
            source=source,
            time=time,
            resolution=resolution,
            variable_name=variable_name,
            normalize_by_area=normalize_by_area,
            zero_is_value=zero_is_value
        )

        if verbose:
            global_summary_stats = utils.dataframe_stats_point(dataframe=points_gdf, agg_column=agg_column, agg_function=agg_function)
            print(f"Global stats before gridding : {global_summary_stats:.2f}")
            global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, normalize_by_area=normalize_by_area, resolution=resolution)
            print(f"Global stats after gridding: {global_gridded_stats:.2f}")
    
    if output_directory or output_filename:
        # Set output directory
        output_directory = (output_directory or os.getcwd()).rstrip(os.sep) + os.sep
        # Set base filename
        base_filename = os.path.splitext(os.path.basename(point_data))[0] if isinstance(point_data, (str, bytes, os.PathLike)) else "gridded_points"
        # Set output filename
        output_filename = output_filename or base_filename
        # save the xarray dataset
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    return ds

def line_2_grid(line_data, variable_name='variable', long_name='variable', units="meter/grid-cell", source=None, time=None, resolution=1, agg_column=None, agg_function="sum", attr_field=None, output_directory=None, output_filename=None, normalize_by_area=False, zero_is_value=False, verbose=False):
    
    """
    Converts line data from a shapefile or GeoDataFrame into a gridded netCDF dataset.

    Parameters
    ----------
    - line_data : GeoDataFrame or str. Input lines data to be gridded. Can be either a GeoDataFrame or a path to a line/polyline shapefile (.shp).
    - variable_name : str, optional. Name of the variable to include in the netCDF attributes metadata. Defaults to:
        - The unique entries in the `attr_field` column if specified.
        - The input filename without extension if `attr_field` and `variable_name` are not specified.
    - long_name : str, optional. A descriptive name for the variable, added to the netCDF metadata. Behaves the same as `variable_name` if
        `attr_field` is specified. Defaults to the input filename without extension if unspecified.
    - units : str, optional. Units of the data variable to include in the netCDF metadata. Default is "meter/grid-cell".
    - source : str, optional. String describing the original source of the input data. This will be added to the netCDF metadata.
    - time : str, optional. Time dimension for the output netCDF. If specified, the output will include a time dimension with the
        value provided. Default is None (spatial, 2D netCDF output).
    - resolution : float, optional. Desired resolution for the grid cells in the output dataset. Default is 1 degree.
    - agg_column : str, optional. Column name in the shapefile or GeoDataFrame specifying the values to aggregate in each grid cell.
        Defaults to summing the lengths of intersected lines per grid cell.
    - agg_function : str, optional. Aggregation method for combining values in each grid cell. Options include:
        - 'sum' (default): Sums all line values.
        - 'max': Takes the maximum value.
        - 'min': Takes the minimum value.
        - 'std': Computes the standard deviation.
    - attr_field : str, optional. Column name in the shapefile or GeoDataFrame specifying the variable names for multiple data types.
    - output_directory : str, optional. Directory where the output NetCDF file will be saved. If None, but output_filename is True, the file will be saved in the current working directory.
    - output_filename : str, optional. Name of the output NetCDF file (without the `.nc` extension). If not provided:
        - Uses the input shapefile name if a shapefile path is given.
        - Saves as `"gridded_lines.nc"` if a GeoDataFrame is provided as input.
    - normalize_by_area : bool, optional. If True, normalizes the variable in each grid cell by the area of the grid cell (e.g., converts to value per square meter). Default is False.
    - zero_is_value : bool, optional.   If True, treats zero values as valid data rather than as no-data. Default is False.
        If True, treats zero values as valid data rather than as no-data. Default is False.
    - verbose : bool, optional. If True, prints information about the process, such as global sum of values before and after gridding. Default is False.

    Returns
    -------
    - xarray.Dataset. Transformed dataset with gridded data derived from the input line data.

    Notes
    -----
    - The function supports input in the form of a shapefile or GeoDataFrame containing line data.
    - Line lengths are calculated and aggregated based on the specified `agg_column` and `agg_function`.
    - If lines intersect a grid boundary, their contributions are divided proportionally among the intersected grid cells.
    - The function creates a netCDF file, where data variables are aggregated and stored with metadata.
    
    Example
    -------
    >>> line_2_grid(line_data=shapefile_path, 
    ...             variable_name="roads", 
    ...             long_name="Roads Length", 
    ...             units="meter/grid-cell", 
    ...             source="OpenStreetMap",  
    ...             resolution=1,
    ...             agg_function="sum", 
    ...             verbose=True)
    ... )
        
    """

    # Determine if input is a path (string or Path) or a GeoDataFrame
    if isinstance(line_data, (str, bytes, os.PathLike)):
        if verbose:
            print("Reading shapefile from path...")
        lines_gdf = gpd.read_file(line_data)
    elif isinstance(line_data, gpd.GeoDataFrame):
        lines_gdf = line_data
    else:
        raise TypeError("Input must be a GeoDataFrame or a shapefile path (string or Path).")

    # create gridded polygon
    polygons_gdf = create.create_gridded_polygon(resolution=resolution, out_polygon_path=None, grid_area=False)
    
    if attr_field is not None:
        unique_rows = lines_gdf[attr_field].unique().tolist()
        dataset_list = []
        
        for filter_var in unique_rows:
            # Filter the GeoDataFrame
            filtered_gdf = lines_gdf[lines_gdf[attr_field] == filter_var].copy()
            joined_gdf = utils.line_intersect(polygons_gdf, filtered_gdf, agg_column=agg_column, agg_function=agg_function)

            # Determine agg_column, long_name, and units for the current iteration
            current_agg_column = agg_column or f"length_{agg_function.lower()}"
            current_long_name = utils.reverse_replace_special_characters(filter_var)
            current_units = utils.determine_units_line(units, normalize_by_area)

            # Convert joined GeoDataFrame to xarray dataset
            ds_var = utils.gridded_poly_2_xarray(
                polygon_gdf=joined_gdf,
                grid_value=current_agg_column,
                long_name=current_long_name,
                units=current_units,
                source=source,
                time=time,
                resolution=resolution,
                variable_name=filter_var,
                normalize_by_area=normalize_by_area,
                zero_is_value=zero_is_value
            )

            # Print or process verbose information
            if verbose:
                global_summary_stats = utils.dataframe_stats_line(dataframe=filtered_gdf, agg_column=agg_column, agg_function=agg_function)
                print(f"Global stats of {filter_var} before gridding : {global_summary_stats:.2f} km.")
                var_name = utils.replace_special_characters(filter_var)
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds_var, variable_name=var_name, normalize_by_area=normalize_by_area, resolution=resolution) * 1e-3
                print(f"Global stats of {filter_var} after gridding: {global_gridded_stats:.2f} km.")

            print("\n")
            dataset_list.append(ds_var)
        
        # Merge all datasets from different filtered GeoDataFrames
        ds = xr.merge(dataset_list)
        
    else:
        joined_gdf = utils.line_intersect(polygons_gdf, lines_gdf, agg_column=agg_column, agg_function=agg_function)

        # Determine agg_column, long_name, and units
        agg_column = agg_column or "length_m"
        long_name = utils.determine_long_name_line(long_name, agg_column, variable_name)
        units = utils.determine_units_line(units, normalize_by_area)
        ds = utils.gridded_poly_2_xarray(
            polygon_gdf=joined_gdf,
            grid_value=agg_column,
            long_name=long_name,
            units=units,
            source=source,
            time=time,
            resolution=resolution,
            variable_name=variable_name,
            normalize_by_area=normalize_by_area,
            zero_is_value=zero_is_value
        )
        
        if verbose:
            if agg_column == "length_m":
                global_summary_stats = utils.dataframe_stats_line(dataframe=lines_gdf, agg_column=agg_column, agg_function=agg_function)
                print(f"Global stats before gridding : {global_summary_stats:.2f} km.")
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, agg_column=agg_column, normalize_by_area=normalize_by_area, resolution=resolution) * 1e-3
                print(f"Global stats after gridding: {global_gridded_stats:.2f} km.")
            else:
                global_summary_stats = utils.dataframe_stats_line(dataframe=lines_gdf, agg_column=agg_column, agg_function=agg_function)
                print(f"Global stats before gridding : {global_summary_stats:.2f}.")
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, agg_column=agg_column, normalize_by_area=normalize_by_area, resolution=resolution)
                print(f"Global stats after gridding: {global_gridded_stats:.2f}.")
    
    if output_directory or output_filename:
        # Set output directory
        output_directory = (output_directory or os.getcwd()).rstrip(os.sep) + os.sep
        # Set base filename
        base_filename = os.path.splitext(os.path.basename(line_data))[0] if isinstance(line_data, (str, bytes, os.PathLike)) else "gridded_lines"
        # Set output filename
        output_filename = output_filename or base_filename
        # save the xarray dataset
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    return ds

def poly_2_grid(polygon_data, variable_name='variable', long_name='variable', units="m2/grid-cell", source=None, time=None, resolution=1, attr_field=None, fraction=False, agg_function="sum", output_directory=None, output_filename=None, normalize_by_area=False, zero_is_value=False, verbose=False):

    """
    Converts polygon data from a shapefile or GeoDataFrame into a gridded netCDF dataset.

    Parameters
    ----------
    - polygon_data : GeoDataFrame or str. Input polygons data to be gridded. Can be either a GeoDataFrame or a path to a polygons shapefile (.shp).
    - variable_name : str, optional. Name of the variable to include in the netCDF attributes metadata. Defaults to:
        - The unique entries in the `attr_field` column if specified.
        - The input filename without extension if `attr_field` and `variable_name` are not specified.
    - long_name : str, optional. A descriptive name for the variable, added to the netCDF metadata. Behaves the same as `variable_name` if
        `attr_field` is specified. Defaults to the input filename without extension if unspecified.
    - units : str, optional. Units of the data variable to include in the netCDF metadata. Default is "m2/grid-cell".
    - source : str, optional. String describing the original source of the input data. This will be added to the netCDF metadata.
    - time : str, optional. Time dimension for the output netCDF. If specified, the output will include a time dimension with the
        value provided. Default is None (spatial, 2D netCDF output).
    - resolution : float, optional. Desired resolution for the grid cells in the output dataset. Default is 1 degree.
    - attr_field : str, optional. Column name in the shapefile or GeoDataFrame specifying the variable names for multiple data types.
    - fraction : bool, optional. If True, calculates the fraction of each polygon within each grid cell. The output values will range from 0 to 1. Default is False.
    - agg_function : str, optional. Aggregation method for combining values in each grid cell. Default is 'sum'. Options include:
        - 'sum': Sum of values.
        - 'max': Maximum value.
        - 'min': Minimum value.
        - 'std': Standard deviation.
    - output_directory : str, optional. Directory where the output NetCDF file will be saved. If None, but output_filename is True, the file will be saved in the current working directory.
    - output_filename : str, optional. Name of the output NetCDF file (without the `.nc` extension). If not provided:
        - Uses the input shapefile name if a shapefile path is given.
        - Saves as `"gridded_polygons.nc"` if a GeoDataFrame is provided as input.
    - normalize_by_area : bool, optional. If True, normalizes the grid values by area (e.g., converts to value per square meter). Default is False.
    - zero_is_value : bool, optional. If True, treats zero values as valid data rather than as no-data. Default is False.
    - verbose : bool, optional. If True, prints information about the process, such as global sum of values before and after gridding. Default is False.    

    Returns
    -------
    - xarray.Dataset. Transformed dataset with gridded data derived from the input polygon data.

    Notes
    -----
    - The function supports input in the form of a shapefile or GeoDataFrame containing polygon data.
    - Polygon areas are calculated and aggregated based on the specified `attr_field` and `agg_function`.
    - If the `fraction` parameter is True, the fraction of each polygon in each grid cell will be computed, with values ranging from 0 to 1.
    - The function creates a netCDF file, where data variables are aggregated and stored with metadata.

    Example
    -------
    >>> poly_2_grid(polygon_data=shapefile_path, 
    ...             units="fraction", 
    ...             source="The new global lithological map database GLiM", 
    ...             resolution=1, 
    ...             attr_field="Short_Name", 
    ...             fraction="yes", 
    ...             verbose=True
    ... )
        
    """

    # Determine if input is a path (string or Path) or a GeoDataFrame
    if isinstance(polygon_data, (str, bytes, os.PathLike)):
        if verbose:
            print("Reading shapefile from path...")
        poly_gdf = gpd.read_file(polygon_data)
    elif isinstance(polygon_data, gpd.GeoDataFrame):
        poly_gdf = polygon_data
    else:
        raise TypeError("Input must be a GeoDataFrame or a shapefile path (string or Path).")

    # create gridded polygon
    polygons_gdf = create.create_gridded_polygon(resolution=resolution, out_polygon_path=None, grid_area=False)
    
    if attr_field is not None:
        unique_rows = poly_gdf[attr_field].unique().tolist()
        dataset_list = []
        
        for filter_var in unique_rows:
            
            # Filter the GeoDataFrame
            filtered_gdf = poly_gdf[poly_gdf[attr_field] == filter_var].copy()
            # Reset the index to ensure sequential indexing
            filtered_gdf.reset_index(drop=True, inplace=True)

            # Determine agg_column, long_name, and units for the current iteration
            grid_value = "frac" if fraction else "in_area"
            current_long_name = utils.reverse_replace_special_characters(filter_var)
            current_units = utils.determine_units_poly(units, normalize_by_area, fraction)

            # Convert GeoDataFrame to xarray dataset
            ds_var = utils.poly_intersect(poly_gdf=filtered_gdf,
                                            polygons_gdf=polygons_gdf, 
                                            variable_name=filter_var, 
                                            long_name=current_long_name,
                                            units=current_units,
                                            source=source,
                                            time=time,
                                            resolution=resolution,
                                            agg_function=agg_function, 
                                            fraction=fraction,
                                            normalize_by_area=normalize_by_area,
                                            zero_is_value=zero_is_value)

            # Print or process verbose information
            if verbose:
                global_summary_stats = utils.dataframe_stats_poly(dataframe=filtered_gdf, agg_function=agg_function)
                print(f"Global stats of {filter_var} before gridding : {global_summary_stats:.2f} km2.")
                filter_var = utils.replace_special_characters(filter_var)
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds_var, variable_name=filter_var, agg_column=grid_value,
                                                              normalize_by_area=True, resolution=resolution) * 1e-6
                print(f"Global stats of {filter_var} after gridding: {global_gridded_stats:.2f} km2.")

            print("\n")
            dataset_list.append(ds_var)
        
        # Merge all datasets from different filtered GeoDataFrames
        ds = xr.merge(dataset_list)
        
    else:
        
        # Determine agg_column, long_name, and units
        grid_value = "frac" if fraction else "in_area"
        long_name = utils.determine_long_name_poly(variable_name, long_name, agg_function)
        units = utils.determine_units_poly(units, normalize_by_area, fraction)
        
        # Convert GeoDataFrame to xarray dataset
        ds = utils.poly_intersect(poly_gdf=poly_gdf,
                                        polygons_gdf=polygons_gdf, 
                                        variable_name=variable_name, 
                                        long_name=long_name,
                                        units=units,
                                        source=source,
                                        time=time,
                                        resolution=resolution,
                                        agg_function=agg_function, 
                                        fraction=fraction,
                                        normalize_by_area=normalize_by_area,
                                        zero_is_value=zero_is_value)

        if verbose:
            global_summary_stats = utils.dataframe_stats_poly(dataframe=poly_gdf, agg_function=agg_function)
            print(f"Global stats before gridding : {global_summary_stats:.2f} km2.")
            variable_name = utils.replace_special_characters(variable_name)
            if fraction:
                normalize_by_area = True
            global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, agg_column=grid_value,
                                                              normalize_by_area=normalize_by_area, resolution=resolution) * 1e-6
            print(f"Global stats after gridding: {global_gridded_stats:.2f} km2.")
    
    if output_directory or output_filename:
        # Set output directory
        output_directory = (output_directory or os.getcwd()).rstrip(os.sep) + os.sep
        # Set base filename
        base_filename = os.path.splitext(os.path.basename(polygon_data))[0] if isinstance(polygon_data, (str, bytes, os.PathLike)) else "gridded_polygons"
        # Set output filename
        output_filename = output_filename or base_filename
        # save the xarray dataset
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    return ds  

def grid_2_grid(raster_data, agg_function, variable_name, long_name, units="value/grid-cell", source=None, time=None, resolution=1, netcdf_variable=None, output_directory=None, output_filename=None, padding="symmetric", zero_is_value=False, normalize_by_area=False, verbose=False):  

    """
    Converts raster data (TIFF or netCDF) into a re-gridded xarray dataset.

    Parameters
    ----------
    - raster_data : str. Path to the input raster data file. This can be a string path to a TIFF (.tif) file, a string path to a NetCDF (.nc or .nc4) file or An already loaded xarray.Dataset object.
        - If `raster_data` is a NetCDF file or an xarray.Dataset, the `netcdf_variable` parameter must also be provided to specify which variable to extract.
    - agg_function : str. Aggregation method to apply when re-gridding. Supported values are 'SUM', 'MEAN', or 'MAX'.
    - variable_name : str. Name of the variable to include in the output dataset.
    - long_name : str. Descriptive name for the variable.
    - units : str, optional. Units for the variable. Default is "value/grid-cell".
    - source : str, optional. Source information for the dataset. Default is None.
    - time : str or None, optional. Time stamp or identifier for the data. Default is None.
    - resolution : int or float, optional. Desired resolution of the grid cells in degree in the output dataset. Default is 1.
    - netcdf_variable : str, optional. Name of the variable to extract from the netCDF file, if applicable. Required for netCDF inputs.
    - output_directory : str, optional. Directory where the output NetCDF file will be saved. If None, but output_filename is True, the file will be saved in the current working directory.
    - output_filename : str, optional. Name of the output NetCDF file (without the `.nc` extension). If not provided:
        - Uses `variable_name` if it is specified.
        - Defaults to `regridded.nc` if none of the above are provided.
    - padding : str, optional. Padding strategy ('symmetric' or 'end').
    - zero_is_value : bool, optional. Whether to treat zero values as valid data rather than as no-data. Default is False.
    - normalize_by_area : bool, optional. Whether to normalize grid values by area (e.g., convert to value per square meter). Default is False.
    - verbose : bool, optional. If True, prints the global sum of values before and after re-gridding. Default is False.

    Returns
    -------
    - xarray.Dataset. Re-gridded xarray dataset containing the processed raster data.

    Notes
    -----
    This function supports raster data in TIFF or netCDF format and performs re-gridding based on 
    the specified `agg_function`. The output dataset will include metadata such as the variable name, 
    long name, units, and optional source and time information.
    
    Example
    -------
    >>> grid_2_grid(raster_path=pop_path, 
    ...             agg_function="sum", 
    ...             variable_name="population_count", 
    ...             long_name="Total Population", 
    ...             units="people per grid", 
    ...             source="WorldPop", 
    ...             resolution=1, 
    ...             time="2020-01-01", 
    ...             verbose="yes"
    ... )
    """

    # Determine the file extension
    if isinstance(raster_data, (str, bytes, os.PathLike)):
        file_extension = os.path.splitext(raster_data)[1].lower()
    elif isinstance(raster_data, xr.Dataset):
        file_extension = ".nc"

    if file_extension == ".tif":
        if verbose:
            print("Reading the tif file.")
        # Convert TIFF data to a re-gridded dataset
        ds = utils.tif_2_ds(input_raster=raster_data, agg_function=agg_function, variable_name=variable_name, 
                      long_name=long_name, units=units, source=source, resolution=resolution, time=time, padding=padding,
                      zero_is_value=zero_is_value, normalize_by_area=normalize_by_area, verbose=verbose)
    
    elif file_extension == ".nc" or file_extension == ".nc4":
        if verbose:
            print("Reading the nc file.")
        # Convert netCDF to TIFF
        netcdf_tif_path, temp_path = utils.netcdf_2_tif(raster_data=raster_data, netcdf_variable=netcdf_variable, time=time)
        # Convert netCDF data to a re-gridded dataset
        ds = utils.tif_2_ds(input_raster=netcdf_tif_path, agg_function=agg_function, variable_name=variable_name, 
                      long_name=long_name, units=units, source=source, resolution=resolution, time=time, padding=padding,
                      zero_is_value=zero_is_value, normalize_by_area=normalize_by_area, verbose=verbose)
        # delete temp folder
        utils.delete_temporary_folder(temp_path)
    else:
        # Print an error message for unrecognized file types
        print("Error: File type is not recognized. File type should be either TIFF or netCDF file.")

    if output_directory or output_filename:
        # Set output directory
        output_directory = (output_directory or os.getcwd()).rstrip(os.sep) + os.sep
        # Set base filename
        base_filename = variable_name or "regridded"
        # Set output filename
        output_filename = output_filename or base_filename
        # save the xarray dataset
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    
    if verbose:
        print("Re-gridding completed!")
    return ds

def table_2_grid(surrogate_data, surrogate_variable, tabular_data, tabular_column, variable_name=None, long_name=None, units="value/grid-cell", source=None, time=None, output_directory=None, output_filename=None, zero_is_value=False, normalize_by_area=False, eez=False, verbose=False):
    """
    Convert tabular data to a gridded dataset by spatially distributing values based on a NetCDF variable and a tabular column.

    Parameters:
    -----------
    - surrogate_data : xarray.Dataset or str. xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded 
        into an xarray.Dataset. The dataset must include the variable specified in `surrogate_variable`.
    - surrogate_variable : str. Variable name in the NetCDF or xarray dataset used for spatial distribution.
    - tabular_data : pandas.DataFrame or str. Tabular dataset as a pandas DataFrame or a path to a CSV file. If a file path is provided, it will be 
        automatically loaded into a DataFrame. The data must include a column named "ISO3" representing country codes. 
        If not present, use the `add_iso3_column` utility function to convert country names to ISO3 codes.     
    - tabular_column : str. Column name in the tabular dataset with values to be spatially distributed.
    - variable_name : str, optional. Name of the variable. Default is None.
    - long_name : str, optional. A long name for the variable. Default is None.
    - units : str, optional. Units of the variable. Default is 'value/grid'.
    - source : str, optional. Source information, if available. Default is None.
    - time : str, optional. Time information for the dataset.
    - output_directory : str, optional. Directory where the output NetCDF file will be saved. If None, but output_filename is True, the file will be saved in the current working directory.
    - output_filename : str, optional. Name of the output NetCDF file (without the `.nc` extension). If not provided:
        - Uses `variable_name` if it is specified.
        - Falls back to `long_name` or `tabular_column` if `variable_name` is not given.
        - Defaults to `gridded_table.nc` if none of the above are provided.
    - zero_is_value: bool, optional. If the value is True, then the function will treat zero as an existent value and 0 values will be considered while calculating mean and STD.
    - normalize_by_area : bool, optional. Whether to normalize grid values by area (e.g., convert to value per square meter). Default is False.
    - eez : bool, optional. If set to True, the function converts the jurisdictional Exclusive Economic Zone (EEZ) values to a spatial grid.
    - verbose: bool, optional. If True, the global gridded sum of before and after re-gridding operation will be printed. If any jurisdiction where surrogate variable is missing and tabular data is evenly distributed over the jurisdiction, the ISO3 codes of evenly distributed countries will also be printed.

    Returns:
    --------
    - xarray.Dataset. Resulting gridded dataset after spatial distribution of tabular values.

    Example
    -------
    >>> table_2_grid(surrogate_data=netcdf_file_path, 
    ...             surrogate_variable="railway_length", 
    ...             tabular_data=csv_file_path, 
    ...             tabular_column="steel", 
    ...             variable_name="railtract_steel", 
    ...             long_name="'Railtrack Steel Mass'", 
    ...             units="g m-2", 
    ...             source="Matitia (2022)", 
    ...             normalize_by_area="yes", 
    ...             verbose="yes"
    ... )
    """
    
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(surrogate_data, (str, bytes, os.PathLike)):
        input_ds = xr.open_dataset(surrogate_data)
    elif isinstance(surrogate_data, xr.Dataset):
        input_ds = surrogate_data
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")

    # Load tabular_data (either path or pandas.DataFrame)
    if isinstance(tabular_data, (str, bytes, os.PathLike)):
        input_df = pd.read_csv(tabular_data)
    elif isinstance(tabular_data, pd.DataFrame):
        input_df = tabular_data
    else:
        raise TypeError("`tabular_data` must be a pandas.DataFrame or a path to a CSV file.")
    
    if variable_name is None:
        variable_name = long_name if long_name is not None else tabular_column

    if long_name is None:
        long_name = variable_name if variable_name is not None else tabular_column

    # check the netcdf resolution
    resolution = abs(float(input_ds['lat'].diff('lat').values[0]))
    resolution_str = str(resolution)

    if time:
        # check and convert ISO3 based on occupation or previous control, given a specific year
        input_df = utils.convert_iso3_by_year(df=input_df, year=time)
    
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")
    if eez:
        country_ds = xr.open_dataset(os.path.join(data_dir, "eezs.1deg.nc"))
        input_ds = input_ds.copy()
        input_ds[surrogate_variable] = input_ds[surrogate_variable].fillna(0)
    else:
        # check and print dataframe's iso3 with country fraction dataset
        utils.check_iso3_with_country_ds(input_df, resolution_str)
        
        if resolution_str == "1" or resolution_str == "1.0":
            country_ds = xr.open_dataset(os.path.join(data_dir, "country_fraction.1deg.2000-2023.a.nc"))
            input_ds = input_ds.copy()
            input_ds[surrogate_variable] = input_ds[surrogate_variable].fillna(0)

        elif resolution_str == "0.5":
            country_ds = xr.open_dataset(os.path.join(data_dir, "country_fraction.0_5deg.2000-2023.a.nc"))
            input_ds = input_ds.copy()
            input_ds[surrogate_variable] = input_ds[surrogate_variable].fillna(0)
            
        elif resolution_str == "0.25":
            country_ds = xr.open_dataset(os.path.join(data_dir, "country_fraction.0_25deg.2000-2023.a.nc"))
            input_ds = input_ds.copy()
            input_ds[surrogate_variable] = input_ds[surrogate_variable].fillna(0)
        else:
            raise ValueError("Please re-grid the netcdf file to 1, 0.5 or 0.25 degree.")

    input_ds, country_ds, a = utils.adjust_datasets(input_ds, country_ds, time)
    print(f"Distributing {variable_name} onto {surrogate_variable}.")

    new_ds = create.create_new_ds(input_ds, tabular_column, country_ds, surrogate_variable, input_df, verbose)

    for var_name in new_ds.data_vars:
        a += np.nan_to_num(new_ds[var_name].to_numpy())

    da = xr.DataArray(a, coords={'lat': input_ds['lat'], 'lon': input_ds['lon']}, dims=['lat', 'lon'])

    if units == 'value/grid-cell':
        units = 'value m-2'

    ds = utils.da_to_ds(da, variable_name, long_name, units, source=source, time=time, resolution=resolution,
                        zero_is_value=zero_is_value, normalize_by_area=normalize_by_area)
    
    if verbose:
        print(f"Global sum of jurisdictional dataset : {input_df[[tabular_column]].sum().item()}")
        global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, agg_column=None, normalize_by_area=normalize_by_area, resolution=resolution)
        print(f"Global stats after gridding: {global_gridded_stats:.2f}")

    if output_directory or output_filename:
        # Set output directory
        output_directory = (output_directory or os.getcwd()).rstrip(os.sep) + os.sep
        # Set base filename
        base_filename = variable_name or "gridded_table"
        # Set output filename
        output_filename = output_filename or base_filename
        # save the xarray dataset
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)

    return ds

def grid_2_table(grid_data, variables=None, time=None, grid_area=None, resolution=1, aggregation=None, agg_function='sum', verbose=False):
    """
    Process gridded data from an xarray Dataset to generate tabular data for different jurisdictions.

    Parameters:
    -----------
    - grid_data : xarray.Dataset or str. xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variables : str, optional. Variables name to be processed. It can either be one variable or list of variables. If None, all variables in the dataset (excluding predefined ones) will be considered.
    - time : str, optional. Time slice for data processing. If provided, the nearest time slice is selected. If None, a default time slice is used.
    - resolution : float, optional. Resolution of gridded data in degree. Default is 1 degree.
    - grid_area : str, optional. Indicator to consider grid area during processing. If 'YES', the variable is multiplied by grid area.
    - aggregation : str, optional. Aggregation level for tabular data. If 'region_1', 'region_2', or 'region_3', the data will be aggregated at the corresponding regional level.
    - agg_function : str, optional, default 'sum'. Aggregation method. Options: 'sum', 'mean', 'max', 'min', 'std'.  
    - verbose : bool, optional. If True, the function will print the global sum of values before and after aggregation.

    Returns:
    --------
    df : pandas DataFrame. Tabular data for different jurisdictions, including ISO3 codes, variable values, and optional 'Year' column.
    """

    df = utils.grid_2_table(grid_data=grid_data, variables=variables, time=time, grid_area=grid_area, resolution=resolution, aggregation=aggregation, method=agg_function, verbose=verbose)
    return df

def add_iso3_column(df, column):
    """
    Convert country names in a DataFrame column to their corresponding ISO3 country codes.

    This function reads a JSON file containing country names and their corresponding ISO3 codes, then 
    maps the values from the specified column in the DataFrame to their ISO3 codes based on the JSON data. 
    The resulting ISO3 codes are added as a new column named 'ISO3'.

    Parameters
    ----------
    - df (pandas.DataFrame): The DataFrame containing a column with country names.
    - column (str): The name of the column in the DataFrame that contains country names.

    Returns
    -------
    - pandas.DataFrame: The original DataFrame with an additional 'ISO3' column containing the ISO3 country codes.

    Raises:
    --------
    - FileNotFoundError: If the JSON file containing country mappings cannot be found.
    - KeyError: If the specified column is not present in the DataFrame.

    Example
    -------
    >>> add_iso3_column(df=dataframe, 
    ...                column="Country"
    ... )
    """

    # Convert country names to ISO3
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")
    json_path = os.path.join(data_dir, "Names.json")
    with open(json_path, 'r') as file:
        country_iso3_data = json.load(file)
        # Map the "Country" column to the new "ISO3" column
        df['ISO3'] = df[column].map(country_iso3_data)
        # Print rows where the specified column has NaN values
        nan_iso3 = df[df["ISO3"].isna()]
        iso3_not_found = nan_iso3[column].unique().tolist()
        # Check if the list is not empty before printing
        if iso3_not_found:
            print(f"Country Not Found: {iso3_not_found}")
    return df

def plot_histogram(dataset, variable, time=None, bin_size=30, color='blue', plot_title=None, x_label=None, remove_outliers=False, log_transform=None, output_dir=None, filename=None):
    
    """
    Create a histogram for an array variable in an xarray dataset.
    Optionally remove outliers and apply log transformations.
    
    Parameters:
    - dataset : xarray.Dataset or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variable: str, the name of the variable to plot.
    - time: str, optional, the time slice to plot.
    - bin_size: int, optional, the number of bins in the histogram.
    - color: str, optional, the color of the histogram bars.
    - plot_title: str, optional, the title for the plot.
    - x_label: str, optional, the label for the x-axis.
    - remove_outliers: bool, optional, whether to remove outliers.
    - log_transform: str, optional, the type of log transformation ('log10', 'log', 'log2').
    - output_dir : str, optional, Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    - filename : str, optional, Filename (with extension) for saving the figure. If not provided, the plot is saved as "output_histogram.png".
    
    Returns:
    - None, displays the plot and optionally saves it to a file.

    Example
    -------
    >>> plot_histogram(dataset=dataset, 
    ...                variable="railway_length", 
    ...                bin_size=30, 
    ...                color='blue', 
    ...                plot_title="Histogram of Railway Length"
    ... )
    """
    plot.plot_histogram(dataset, variable, time, bin_size, color, plot_title, x_label, remove_outliers, log_transform, output_dir, filename)
    
def plot_scatter(dataset, variable1, variable2, dataset2=None, time=None, color='blue', x_label=None, y_label=None, plot_title=None, remove_outliers=False, log_transform_1=None, log_transform_2=None, equation=False, output_dir=None, filename=None):
    """
    Create a scatter plot for two variables in an xarray dataset.
    Optionally remove outliers and apply log transformations.
    
    Parameters:
    - variable1 : str, name of the variable to be plotted on the x-axis. Must be present in `dataset`.
    - variable2 : str, name of the variable to be plotted on the y-axis. If `dataset2` is provided, this variable will be extracted from `dataset2`; otherwise, it must exist in `dataset`.
    - dataset : xarray.Dataset or str, the primary dataset or a path to a NetCDF file. This dataset must contain the variable specified by `variable1`, which will be used for the x-axis.
    - dataset2 : xarray.Dataset or str, optional, a second dataset or a path to a NetCDF file containing the variable specified by `variable2` (for the y-axis). If not provided, `dataset` will be used for both variables.
    - time: str, optional, the time slice to plot.
    - color: str, optional, the color map of the scatter plot.
    - x_label: str, optional, the label for the x-axis.
    - y_label: str, optional, the label for the y-axis.
    - plot_title: str, optional, the title for the plot.
    - remove_outliers: bool, optional, whether to remove outliers from the data.
    - log_transform_1: str, optional, the type of log transformation for variable1 ('log10', 'log', 'log2').
    - log_transform_2: str, optional, the type of log transformation for variable2 ('log10', 'log', 'log2').
    - equation : bool, optional, ff True, fits and displays a linear regression equation. 
    - output_dir : str, optional, Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    - filename : str, optional, Filename (with extension) for saving the figure. If not provided, the plot is saved as "output_scatter.png".
    
    Returns:
    - None, displays the plot and optionally saves it to a file.

    Example
    -------
    >>> plot_scatter(dataset=ds_road, 
    ...             variable1="roads_gross", 
    ...             variable2="buildings_gross", 
    ...             dataset2=ds_build, 
    ...             color='blue',
    ...             plot_title="Building vs Road", 
    ...             remove_outliers=True, 
    ...             log_transform_1="log10", 
    ...             log_transform_2="log10"
    ... )
    """
    plot.plot_scatter(dataset, variable1, variable2, dataset2, time, color, x_label, y_label, plot_title, remove_outliers, log_transform_1, log_transform_2, equation, output_dir, filename)
    
def plot_time_series(dataset, variable, agg_function='sum', plot_type='both', color='blue', plot_label='Area Plot', x_label='Year', y_label='Value', plot_title='Time Series Plot', smoothing_window=None, output_dir=None, filename=None):
    """
    Create a line plot and/or area plot for a time series data variable.
    
    Parameters:
    - dataset : xarray.Dataset or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variable: str, the name of the variable to plot.
    - agg_function: str, the operation to apply ('sum', 'mean', 'max', 'std').
    - plot_type: str, optional, the type of plot ('line', 'area', 'both'). Default is 'both'.
    - color: str, optional, the color of the plot. Default is 'blue'.
    - plot_label: str, optional, the label for the plot. Default is 'Area Plot'.
    - x_label: str, optional, the label for the x-axis. Default is 'Year'.
    - y_label: str, optional, the label for the y-axis. Default is 'Value'.
    - plot_title: str, optional, the title of the plot. Default is 'Time Series Plot'.
    - smoothing_window: int, optional, the window size for rolling mean smoothing.
    - output_dir : str, optional, Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    - filename : str, optional, Filename (with extension) for saving the figure. If not provided, the plot is saved as "output_time_series.png".
    
    Returns:
    - None, displays the plot and optionally saves it to a file.

    Example
    -------
    >>> plot_time_series(variable="buildings_gross", 
    ...                dataset=ds_build, 
    ...                agg_function='sum', 
    ...                plot_type='both', 
    ...                color='blue', 
    ...                x_label='Year', 
    ...                y_label='Value', 
    ...                plot_title='Time Series Plot'
    ... )    
    """
    
    plot.plot_time_series(dataset, variable, agg_function, plot_type, color, plot_label, x_label, y_label, plot_title, smoothing_window, output_dir, filename)

def plot_hexbin(dataset, variable1, variable2, dataset2=None, time=None, color='pink_r', grid_size=30, x_label=None, y_label=None, plot_title=None, remove_outliers=False, log_transform_1=None, log_transform_2=None, output_dir=None, filename=None):
    
    """
    Create a hexbin plot for two variables in an xarray dataset.

    Parameters:
    - dataset : xarray.Dataset or str, the primary dataset or a path to a NetCDF file. This dataset must contain the variable specified by `variable1`, which will be used for the x-axis.
    - variable1 : str, name of the variable to be plotted on the x-axis. Must be present in `dataset`.
    - variable2 : str, name of the variable to be plotted on the y-axis. If `dataset2` is provided, this variable will be extracted from `dataset2`; otherwise, it must exist in `dataset`.
    - dataset2 : xarray.Dataset or str, optional, a second dataset or a path to a NetCDF file containing the variable specified by `variable2` (for the y-axis). If not provided, `dataset` will be used for both variables.
    - time: str, optional, the time slice to plot.
    - color: str, optional, the color map of the hexbin plot.
    - grid_size: int, optional, the number of hexagons in the x-direction.
    - x_label: str, optional, the label for the x-axis.
    - y_label: str, optional, the label for the y-axis.
    - plot_title: str, optional, the title for the plot.
    - remove_outliers: bool, optional, whether to remove outliers from the data.
    - log_transform_1: str, optional, the type of log transformation for variable1 ('log10', 'log', 'log2').
    - log_transform_2: str, optional, the type of log transformation for variable2 ('log10', 'log', 'log2').
    - output_dir : str, optional, Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    - filename : str, optional, Filename (with extension) for saving the figure. If not provided, the plot is saved as "output_hexbin.png".
    
    Returns:
    - None, displays the map and optionally saves it to a file.
    Example
    -------
    >>> plot_hexbin(dataset=ds_road, 
    ...             variable1="roads_gross", 
    ...             variable2="buildings_gross", 
    ...             dataset2=ds_build, 
    ...             color='blue', 
    ...             plot_title="Building vs Road"
    ... )
    """
    
    plot.plot_hexbin(dataset, variable1, variable2, dataset2, time, color, grid_size, x_label, y_label, plot_title, remove_outliers, log_transform_1, log_transform_2, output_dir, filename)
    
def plot_map(dataset, variable, time=None, depth=None, color='hot_r', title='', label='', vmin=None, vmax=None, extend_min=None, extend_max=None, levels=10, out_bound=True, remove_ata=False, output_dir=None, filename=None, show=True):
    
    """
    Plots a 2D map of a variable from an xarray Dataset or NetCDF file with customizable colorbar, projection, and map appearance.

    Parameters
    ----------
    - dataset : xarray.Dataset. or str. xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variable : str. Name of the variable in the xarray Dataset to plot.
    - color : str, default 'hot_r'. Matplotlib colormap name for the plot (discrete color scale).
    - title : str, default ''. Title of the map.
    - label : str, default ''. Label for the colorbar.
    - time: str, optional, the time slice to plot.
    - depth: str, optional, the depth slice to plot.
    - vmin : float, optional. Minimum data value for the colorbar range. If not provided, the minimum of the variable is used.
    - vmax : float, optional. Maximum data value for the colorbar range. If not provided, the maximum of the variable is used.
    - extend_min : bool or None, default None. If True, includes values below `vmin` in the first color class and shows a left arrow on the colorbar.
    - extend_max : bool or None, default None. If True, includes values above `vmax` in the last color class and shows a right arrow on the colorbar.
    - levels : int or list of float, default 10. Either the number of color intervals or a list of explicit interval boundaries.
    - out_bound : bool, default True. Whether to display the outer boundary (spine) of the map projection.
    - remove_ata : bool, default False. If True, removes Antarctica from the map by excluding data below 60°S latitude.
    - output_dir : str, optional. Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    - filename : str, optional. Filename (with extension) for saving the figure. If not provided, the plot is saved as "output_plot.png".
    - show : bool, True. Whether or not show the map

    Notes
    -----
    - If both `extend_min` and `extend_max` are False, the dataset is clipped strictly within [vmin, vmax].
    - The colorbar will use arrows to indicate out-of-bound values only if `extend_min` or `extend_max` is True.
    - Tick formatting on the colorbar is:
        - Two decimal places if (vmax - vmin) <= 10.
    - If `remove_ata` is True, the colorbar is placed slightly higher to avoid overlap with the map.
        
    Returns:
    - Axes class of the map, optionally displays the map and saves it to a file.
    - Formats a discrete colorbar by always labeling all bin boundaries, automatically using scientific notation for large or small values while avoiding unnecessary ×10⁰ scaling.

    Example
    -------
    >>> plot_map(
    ...     dataset=ds.isel(time=-1),
    ...     variable='npp',
    ...     vmin=0,
    ...     vmax=1200,
    ...     extend_max=True,
    ...     color='Greens',
    ...     levels=10,
    ...     remove_ata=True,
    ...     title='Net Primary Productivity',
    ...     label='gC/m²/year',
    ...     filename='npp_map.png'
    ... )
    """
    
    ax = plot.plot_map(dataset=dataset, variable=variable, time=time, depth=depth, color=color, title=title, label=label,
             vmin=vmin, vmax=vmax, extend_min=extend_min, extend_max=extend_max, levels=levels, 
             out_bound=out_bound, remove_ata=remove_ata, output_dir=output_dir, filename=filename, show=show)
    return ax

def plot_country(tabular_data, column, title="", label="", color='viridis', levels=10, output_dir=None, filename=None, remove_ata=False, out_bound=True, vmin=None, vmax=None, extend_min=None, extend_max=None, show=True):
    """
    Plots a choropleth map of countries using a specified data column and a world shapefile.

    Parameters:
    -----------
    - tabular_data : pandas.DataFrame or str. Input table containing country-level data. Can be either:
        - A pandas DataFrame with the required `column`
        - A string path to a CSV file, which will be automatically read into a DataFrame
    - column : str. Name of the column in the dataframe to visualize.
    - title : str, optional. Title of the map. Default is an empty string.
    - label : str, optional. Label for the colorbar. Default is an empty string.
    - color : str, optional. Name of the matplotlib colormap to use. Default is 'viridis'.
    - levels : int or list of float, optional. Number of color levels (if int) or list of bin edges (if list). Default is 10.
    - remove_ata : bool, optional. Whether to remove Antarctica ('ATA') from the data. Default is False.
    - out_bound : bool, optional. Whether to display map boundaries (spines). Default is True.
    - vmin : float or None, optional. Minimum value for the colormap. If None, calculated from the data.
    - vmax : float or None, optional. Maximum value for the colormap. If None, calculated from the data.
    - extend_min : bool or None, default None. If True, includes values below `vmin` in the first color class and shows a left arrow on the colorbar.
    - extend_max : bool or None, default None. If True, includes values above `vmax` in the last color class and shows a right arrow on the colorbar.
    - output_dir : str, optional. Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    - filename : str, optional. Filename (with extension) for saving the figure. If not provided, the plot is saved as "output_country_plot.png".
    - show : bool, True. Whether or not show the map

    Returns:
    --------
    - None, displays the map and optionally saves it to a file.
    - Formats a discrete colorbar by always labeling all bin boundaries, automatically using scientific notation for large or small values while avoiding unnecessary ×10⁰ scaling.

    Example
    -------
    >>> plot_country(tabular_data="country_data.csv", 
    ...             column="population", 
    ...             title="Population of Countries", 
    ...             label="Population", 
    ...             color='viridis'
    ... )
    """

    ax = plot.plot_country(tabular_data=tabular_data, column=column, title=title, label=label, color=color, levels=levels, output_dir=output_dir, filename=filename, remove_ata=remove_ata, out_bound=out_bound, vmin=vmin, vmax=vmax, extend_min=extend_min, extend_max=extend_max, show=show)
    
    return ax
            
def sum_variables(dataset, variables=None, new_variable_name=None, time=None):

    """
    Sum specified variables in the xarray dataset. If no variables are specified, sum all variables
    except those starting with 'grid_area'. Fill NaNs with zero before summing, and convert resulting
    zeros back to NaNs.
    
    Parameters:
    -----------
    - dataset: xarray.Dataset. or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variables: list of str, the names of the variables to sum. If None, sum all variables except those starting with 'grid_area' and 'land_frac'.
    - new_variable_name: str, optional, the name of the new variable to store the sum.
    - time: optional, a specific time slice to select from the dataset.
    
    Returns:
    --------
    - xarray.Dataset. with the summed variable.

    Example
    -------
    >>> sum_variables(dataset=ds, 
    ...              variables=["roads_gross", "buildings_gross"], 
    ...              new_variable_name="gross_mass"
    ... )
    """
    
    ds = calculate.sum_variables(dataset, variables, new_variable_name, time)
    return ds
    
def subtract_variables(dataset, variable1, variable2, new_variable_name=None, time=None):
    
    """
    Subtract one variable from another in the xarray dataset.
    Fill NaNs with zero before subtracting, and convert resulting zeros back to NaNs.
    
    Parameters:
    -----------
    - dataset: xarray.Dataset. or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variable1: str, the name of the variable to subtract from.
    - variable2: str, the name of the variable to subtract.
    - new_variable_name: str, optional, the name of the new variable to store the result.
    - time: optional, a specific time slice to select from the dataset.
    
    Returns:
    --------
    - xarray.Dataset. with the resulting variable.

    Example
    -------
    >>> subtract_variables(dataset=ds,
    ...                   variable1="precipitation", 
    ...                   variable2="evaporation", 
    ...                   new_variable_name="net_water_gain"
    ... )
    """
    ds = calculate.subtract_variables(variable1, variable2, dataset, new_variable_name, time)
    return ds
    
def divide_variables(dataset,variable1, variable2, new_variable_name=None, time=None):
    """
    Divide one variable by another in the xarray dataset.
    Fill NaNs with zero before dividing, and convert resulting zeros back to NaNs.
    
    Parameters:
    -----------
    - dataset: xarray.Dataset. or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.    
    - variable1: str, the name of the variable to be divided (numerator).
    - variable2: str, the name of the variable to divide by (denominator).
    - new_variable_name: str, optional, the name of the new variable to store the result.
    - time: optional, a specific time slice to select from the dataset.
    
    Returns:
    --------
    - xarray.Dataset. with the resulting variable.

    Example
    -------
    >>> divide_variables(dataset=ds,
    ...                  variable1="road_length", 
    ...                  variable2="grid_area", 
    ...                  new_variable_name="road_density"
    ... )
    """
    ds = calculate.divide_variables(variable1, variable2, dataset, new_variable_name, time)
    return ds
    
def multiply_variables(dataset, variables=None, new_variable_name=None, time=None):
    """
    Multiply specified variables in the xarray dataset. If no variables are specified, multiply all variables.
    Fill NaNs with one before multiplying, and convert resulting ones back to NaNs.
    
    Parameters:
    -----------
    - dataset: xarray.Dataset. or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variables: list of str, the names of the variables to multiply. If None, multiply all variables, excluding the "grid_area" and "land_frac" variables included in the dataset.
    - new_variable_name: str, optional, the name of the new variable to store the product.
    - time: optional, a specific time slice to select from the dataset.
    
    Returns:
    --------
    - xarray.Dataset. with the resulting variable.

    Example
    -------
    >>> multiply_variables(
    ...     dataset=ds,
    ...     variables=["crop_area", "yield_per_hectare"],
    ...     new_variable_name="total_crop_yield"
    ... )
    """
    
    ds = calculate.multiply_variables(dataset, variables, new_variable_name, time)
    
    return ds
    
def average_variables(dataset, variables=None, new_variable_name=None, time=None):
    """
    Average specified variables in the xarray dataset. If no variables are specified, average all variables
    except those starting with 'grid_area'. Fill NaNs with zero before averaging, and convert resulting
    zeros back to NaNs.
    
    Parameters:
    -----------
    - dataset: xarray.Dataset. or str, xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variables: list of str, the names of the variables to average. If None, average all variables except those starting with 'grid_area' and 'land_frac'.
    - new_variable_name: str, optional, the name of the new variable to store the average.
    - time: optional, a specific time slice to select from the dataset.
    
    Returns:
    --------
    - xarray.Dataset. with the averaged variable.

    Example
    -------
    >>> average_variables(dataset=ds, 
    ...                  variables=["roads_gross", "buildings_gross"], 
    ...                  new_variable_name="average_gross"
    ... )
    """
    ds = calculate.average_variables(dataset, variables, new_variable_name, time)
    return ds

def get_netcdf_info(netcdf_file, variable_name=None):
    """
    Extract information about variables and dimensions from a NetCDF dataset.

    Parameters
    ----------
    - netcdf_file : xarray.Dataset or str. xarray dataset or a path to a NetCDF file. If a file path is provided, it will be automatically loaded into an xarray.Dataset.
    - variable_name : str, optional. The prefix or complete name of the variable to filter. If not provided, all variables are included.

    Returns
    -------
    - tuple, A tuple containing lists of dimensions, short names, long names, units, & time values (if 'time' exists).
        
    Example
    -------
    >>> get_netcdf_info(netcdf_file=netcdf_file_path, 
    ...                 variable_name="railway_length"
    ... )
    """

    netcdf_info = get.get_netcdf_info(netcdf_file=netcdf_file, variable_name=variable_name)
    return netcdf_info

def atlas(directory):
    """
    List all NetCDF files in a directory and count the number of variables in each.

    Parameters
    ----------
    directory : str. Path to the directory containing NetCDF files.

    Returns
    -------
    pd.DataFrame. A DataFrame with file names and the number of variables in each file.
    
    Example
    -------
    >>> atlas(directory)
    """
    records = []
    for file in os.listdir(directory):
        if file.endswith(".nc"):
            filepath = os.path.join(directory, file)
            ds = xr.open_dataset(filepath)
            num_vars = len(ds.data_vars)
            ds.close()
            records.append({
                'file_name': file,
                'num_variables': num_vars
            })
    return pd.DataFrame(records)

def info(data):
    """
    Extract metadata for each variable in a NetCDF dataset.

    Parameters
    ----------
    - data : str, os.PathLike, or xarray.Dataset. Path to a NetCDF file or an xarray.Dataset object.

    Returns
    -------
    - pd.DataFrame. A DataFrame containing variable names, long names, units, sources, time range (start and end), time resolution (step), and depth values (if present as a variable).
    
    Example
    -------
    >>> info(netcdf_path)
    """
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(data, (str, bytes, os.PathLike)):
        ds = xr.open_dataset(data)
    elif isinstance(data, xr.Dataset):
        ds = data
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")  

    records = []
    for var_name, da in ds.data_vars.items():
        var_attrs = da.attrs
        # Handle time and depth dimensions if they exist
        time_summary = depth_summary = None

        if 'time' in da.dims:
            if np.issubdtype(da['time'].dtype, np.datetime64):
                time_values = pd.to_datetime(da['time'].values.flatten())
                unique_times = np.unique(time_values)
                time_diffs = np.diff(unique_times)
                time_step = utils.detect_time_step(time_diffs) if len(unique_times) > 1 else None
                time_summary = {
                    'min': pd.to_datetime(unique_times.min()).strftime('%Y-%m-%d'),
                    'max': pd.to_datetime(unique_times.max()).strftime('%Y-%m-%d'),
                    'step': time_step
                }
            else:
                unique_times = np.unique(da['time'].values.flatten())
                time_summary = {
                    'min': int(unique_times.min()),
                    'max': int(unique_times.max()),
                    'step': 'Monthly' if set(unique_times).issubset(set(range(1, 13))) else 'Unknown'
                }

        if 'depth' in da.dims and 'depth' in ds.variables:
            depth_values = ds['depth'].values.flatten()
            unique_depths = np.unique(depth_values)
            depth_summary = {
                'values': unique_depths.tolist()
            }

        records.append({
            'variable': var_name,
            'long_name': var_attrs.get('long_name', 'N/A'),
            'units': var_attrs.get('units', 'N/A'),
            'source': var_attrs.get('source', 'N/A'),
            'time_min': time_summary['min'] if time_summary else None,
            'time_max': time_summary['max'] if time_summary else None,
            'time_step': time_summary['step'] if time_summary else None,
            'depth': depth_summary['values'] if depth_summary else None
        })

    ds.close()
    return pd.DataFrame(records)
