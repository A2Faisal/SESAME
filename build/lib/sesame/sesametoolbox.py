import os
import re
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, LineString, Point
import pyproj
import numpy as np
import xarray as xr
import json

# import local libraries
from . import create
from . import utils
from . import calculate
from . import plot
from . import get

def table_2_grid(netcdf_variable, tabular_column, netcdf_file_path=None, csv_file_path=None, input_ds=None,
                 input_df=None, variable_name=None, long_name=None, units="value/grid-cell", source=None,
                 time=None, output_directory=None, output_filename=None, zero_is_value=None, value_per_area=None, verbose=False):
    """
    Convert tabular data to a gridded dataset by spatially distributing values based on a NetCDF variable and a tabular column.

    Parameters:
    -----------
    netcdf_variable : str
        Variable name in the NetCDF dataset used for spatial distribution.
    tabular_column : str
        Column name in the tabular dataset with values to be spatially distributed.
    netcdf_file_path : str, optional
        a netcdf variable data path as string
    csv_file_path : str, optional
        a tabular file where data is stored based on their jurisdiction or ISO3 code. The csv file must hold a
        column named “ISO3”. If not, then users must use jurisdiction_2_ISO3 function to convert the country
        name to their corresponding ISO3 code.
    input_ds : xarray.Dataset
        Input NetCDF dataset with spatial coordinates.
    input_df : pandas.DataFrame
        Input tabular dataset containing values to be distributed spatially.
    variable_name : str, optional
        Name of the variable. Default is None.
    long_name : str, optional
        A long name for the variable. Default is None.
    units : str, optional
        Units of the variable. Default is 'value/grid'.
    source : str, optional
        Source information, if available. Default is None.
    filename : str, optional
        Additional name for the output raster file. If None, a default name will be used.
    time : str, optional
        Time information for the dataset.
    zero_is_value: str, optional
        If the value is “yes”, then the function will treat zero as an existent value and 0 values will be
        considered while calculating mean and STD.
    value_per_sqm : float, optional
        if input “yes” then the value will be transformed into “value m-2”.
    verbose: bool, optional
        If “yes”, the global gridded sum of before and after re-gridding operation will be printed. If any
        jurisdiction where surrogate variable is missing and tabular data is evenly distributed over the
        jurisdiction, the ISO3 codes of evenly distributed countries will also be printed.

    Returns:
    --------
    ds : xarray.Dataset
        Resulting gridded dataset after spatial distribution of tabular values.
    """
    if netcdf_file_path and csv_file_path:
        input_ds = xr.open_dataset(netcdf_file_path)
        input_df = pd.read_csv(csv_file_path)

    if not isinstance(input_ds, xr.Dataset) and isinstance(input_df, pd.DataFrame):
        raise ValueError("Please provide either netcdf and csv file paths or xarray dataset and pandas dataframe.")

    if variable_name is None:
        variable_name = long_name if long_name is not None else tabular_column

    if long_name is None:
        long_name = variable_name if variable_name is not None else tabular_column

    # check the netcdf resolution
    cell_size = abs(float(input_ds['lat'].diff('lat').values[0]))
    cell_size_str = str(cell_size)

    if time:
        # check and convert ISO3 based on occupation or previous control, given a specific year
        input_df = utils.convert_iso3_by_year(df=input_df, year=time)

    # check and print dataframe's iso3 with country fraction dataset
    utils.check_iso3_with_country_ds(input_df, cell_size_str)
  
    base_directory = os.path.dirname(os.path.abspath(__file__))
    if cell_size_str == "1" or cell_size_str == "1.0":
        country_ds = xr.load_dataset(os.path.join(base_directory, "country_fraction.1deg.2000-2023.a.nc"))
        # Remove surrogate variable if land_frac is 0
        grid_ds = xr.open_dataset(os.path.join(base_directory, "G.land_sea_mask.nc"))
        grid_ds["land_frac"] = grid_ds["land_frac"].where(grid_ds["land_frac"] == 0, 1)
        input_ds = input_ds.copy()
        input_ds[netcdf_variable] = input_ds[netcdf_variable].fillna(0) * grid_ds["land_frac"]

    elif cell_size_str == "0.5":
        country_ds = xr.load_dataset(os.path.join(base_directory, "country_fraction.0_5deg.2000-2023.a.nc")) 
    else:
        raise ValueError("Please re-grid the netcdf file to 1 or 0.5 degree.")

    input_ds, country_ds, a = utils.adjust_datasets(input_ds, country_ds, time)
    print(f"Distributing {variable_name} onto {netcdf_variable}.")

    new_ds = create.create_new_ds(input_ds, tabular_column, country_ds, netcdf_variable, input_df, verbose)

    for var_name in new_ds.data_vars:
        a += np.nan_to_num(new_ds[var_name].to_numpy())

    da = xr.DataArray(a, coords={'lat': input_ds['lat'], 'lon': input_ds['lon']}, dims=['lat', 'lon'])

    # # TODO: move verbose to after creating dataset
    # if verbose:
    #     print(f"Global sum of jurisdictional dataset : {input_df[[tabular_column]].sum().item()}")
    #     print(f"Global sum of gridded dataset : {da.sum().item()}\n")

    if units == 'value/grid-cell':
        units = 'value m-2'

    ds = utils.da_to_ds(da, variable_name, long_name, units, source=source, time=time, cell_size=cell_size,
                        zero_is_value=zero_is_value, value_per_area=value_per_area)
    
    if verbose:
        print(f"Global sum of jurisdictional dataset : {input_df[[tabular_column]].sum().item()}")
        global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, fold_field=None, value_per_area=value_per_area, cell_size=cell_size)
        print(f"Global stats after gridding: {global_gridded_stats:.2f}")

    # save the xarray dataset
    if output_directory:
        if netcdf_file_path:
            base_filename = os.path.splitext(os.path.basename(netcdf_file_path))[0]
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)

    return ds



def grid_2_grid(raster_path, fold_function, variable_name, long_name, units="value/grid-cell", source=None, time=None, cell_size=1, netcdf_variable=None, output_directory=None, 
                 output_filename=None, zero_is_value=None, value_per_area=False, verbose=False):  
    """
    Convert raster data (TIFF or netCDF) to a re-gridded xarray dataset.

    Parameters
    ----------
    raster_path : str
        File path to the input raster data. Should be either a TIFF or netCDF file.
    conversion_type : str
        Type of conversion to perform ('SUM', 'MEAN', or 'MAX').
    short_name : str
        Name of the variable.
    long_name : str
        A long name for the variable.
    units : str
        Units of the variable.
    source : str, optional
        Source information, if available. Default is None.
    time : str or None, optional
        Time associated with the data. Default is None.
    cell_size : int or float, optional
        Desired cell size for the output grid. Default is 1.
    netcdf_variable : str, optional
        Variable in the netCDF data to be converted. Required if the input is a netCDF file.
    cell_registration : str, optional
        Cell registration method for raster datasets. Default is "UPPER_LEFT".
    multidimensional : str or None, optional
        Specify "YES" if dealing with multidimensional netCDF data. Default is None.

    Returns
    -------
    xarray.Dataset
        Re-gridded xarray dataset containing the converted raster data.
    """

    # Determine the file extension
    file_extension = os.path.splitext(raster_path)[1]

    if file_extension == ".tif":
        print("Reading the tif file.")
        # Convert TIFF data to a re-gridded dataset
        ds = utils.tif_2_ds(input_raster=raster_path, fold_function=fold_function, variable_name=variable_name, 
                      long_name=long_name, units=units, source=source, cell_size=cell_size, time=time, 
                      zero_is_value=zero_is_value, value_per_area=value_per_area, verbose=verbose)
    
    elif file_extension == ".nc" or file_extension == ".nc4":
        # Convert netCDF to TIFF
        print("Reading the nc file.")
        netcdf_tif_path = utils.netcdf_2_tif(netcdf_path=raster_path, netcdf_variable=netcdf_variable, time=time)
        # Convert netCDF data to a re-gridded dataset
        ds = utils.tif_2_ds(input_raster=netcdf_tif_path, fold_function=fold_function, variable_name=variable_name, 
                      long_name=long_name, units=units, source=source, cell_size=cell_size, time=time, 
                      zero_is_value=zero_is_value, value_per_area=value_per_area, verbose=verbose)
    else:
        # Print an error message for unrecognized file types
        print("Error: File type is not recognized. File type should be either TIFF or netCDF file.")

    # save the xarray dataset
    if output_directory:
        if raster_path:
            base_filename = os.path.splitext(os.path.basename(raster_path))[0]
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)

    return ds


def poly_2_grid(poly_gdf=None, variable_name=None, long_name=None, units="m2/grid-cell", source=None, time=None, 
                 cell_size=1, attr_field=None, shapefile_path=None, fraction=False, fold_function="sum", output_directory=None, 
                 output_filename=None, value_per_area=False, zero_is_value=False, verbose=False):
    
    if poly_gdf is None and shapefile_path is None:
        raise ValueError("Either 'polygons geodataframe' or 'shapefile directory' must be provided.")
    elif poly_gdf is not None and shapefile_path is not None:
        raise ValueError("Only one of 'polygons geodataframe' or 'shapefile directory' should be provided.")

    if shapefile_path:
        poly_gdf = gpd.read_file(shapefile_path)

    # create gridded polygon
    polygons_gdf = create.create_gridded_polygon(cell_size=cell_size, out_polygon_path=None, grid_area=False)
    
    if attr_field is not None:
        unique_rows = poly_gdf[attr_field].unique().tolist()
        dataset_list = []
        
        for filter_var in unique_rows:
            
            # Filter the GeoDataFrame
            filtered_gdf = poly_gdf[poly_gdf[attr_field] == filter_var].copy()
            # Reset the index to ensure sequential indexing
            filtered_gdf.reset_index(drop=True, inplace=True)

            # Determine fold_field, long_name, and units for the current iteration
            grid_value = "frac" if fraction else "in_area"
            current_long_name = utils.reverse_replace_special_characters(filter_var)
            current_units = utils.determine_units_poly(units, value_per_area, fraction)

            # Convert GeoDataFrame to xarray dataset
            ds_var = utils.poly_intersect(poly_gdf=filtered_gdf,
                                            polygons_gdf=polygons_gdf, 
                                            variable_name=filter_var, 
                                            long_name=current_long_name,
                                            units=current_units,
                                            source=source,
                                            time=time,
                                            cell_size=cell_size,
                                            fold_function=fold_function, 
                                            fraction=fraction,
                                            value_per_area=value_per_area,
                                            zero_is_value=zero_is_value)

            # Print or process verbose information
            if verbose:
                global_summary_stats = utils.dataframe_stats_poly(dataframe=filtered_gdf, fold_function=fold_function)
                print(f"Global stats of {filter_var} before gridding : {global_summary_stats:.2f} km2.")
                filter_var = utils.replace_special_characters(filter_var)
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds_var, variable_name=filter_var, fold_field=grid_value,
                                                              value_per_area=fraction, cell_size=cell_size) * 1e-6
                print(f"Global stats of {filter_var} after gridding: {global_gridded_stats:.2f} km2.")

            print("\n")
            dataset_list.append(ds_var)
        
        # Merge all datasets from different filtered GeoDataFrames
        ds = xr.merge(dataset_list)
        
    else:
        
        # Determine fold_field, long_name, and units
        grid_value = "frac" if fraction else "in_area"
        long_name = utils.determine_long_name_poly(variable_name, long_name, fold_function)
        units = utils.determine_units_poly(units, value_per_area, fraction)
        
        # Convert GeoDataFrame to xarray dataset
        ds = utils.poly_intersect(poly_gdf=poly_gdf,
                                        polygons_gdf=polygons_gdf, 
                                        variable_name=variable_name, 
                                        long_name=long_name,
                                        units=units,
                                        source=source,
                                        time=time,
                                        cell_size=cell_size,
                                        fold_function=fold_function, 
                                        fraction=fraction,
                                        value_per_area=value_per_area,
                                        zero_is_value=zero_is_value)

        if verbose:
            global_summary_stats = utils.dataframe_stats_poly(dataframe=poly_gdf, fold_function=fold_function)
            print(f"Global stats before gridding : {global_summary_stats:.2f} km2.")
            variable_name = utils.replace_special_characters(variable_name)
            global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, fold_field=grid_value,
                                                              value_per_area=fraction, cell_size=cell_size) * 1e-6
            print(f"Global stats after gridding: {global_gridded_stats:.2f} km2.")
    
    # save the xarray dataset
    if output_directory:
        if shapefile_path:
            base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    return ds  
  
    
    
def line_2_grid(lines_gdf=None, variable_name=None, long_name=None, units="meter/grid-cell", source=None, time=None, 
                 cell_size=1, fold_field=None, fold_function="sum", attr_field=None, shapefile_path=None, 
                 output_directory=None, output_filename=None, value_per_area=False, zero_is_value=False, verbose=False):
    
    if lines_gdf is None and shapefile_path is None:
        raise ValueError("Either 'lines geodataframe' or 'shapefile directory' must be provided.")
    elif lines_gdf is not None and shapefile_path is not None:
        raise ValueError("Only one of 'lines geodataframe' or 'shapefile directory' should be provided.")

    # create gridded polygon
    polygons_gdf = create.create_gridded_polygon(cell_size=cell_size, out_polygon_path=None, grid_area=False)
    
    # spatial join
    if shapefile_path:
        lines_gdf = gpd.read_file(shapefile_path)
    
    if attr_field is not None:
        unique_rows = lines_gdf[attr_field].unique().tolist()
        dataset_list = []
        
        for filter_var in unique_rows:
            # Filter the GeoDataFrame
            filtered_gdf = lines_gdf[lines_gdf[attr_field] == filter_var].copy()
            joined_gdf = utils.line_intersect(polygons_gdf, filtered_gdf, fold_field=fold_field, fold_function=fold_function)

            # Determine fold_field, long_name, and units for the current iteration
            current_fold_field = fold_field or f"length_{fold_function.lower()}"
            current_long_name = utils.reverse_replace_special_characters(filter_var)
            current_units = utils.determine_units_line(units, value_per_area)

            # Convert joined GeoDataFrame to xarray dataset
            ds_var = utils.gridded_poly_2_xarray(
                polygon_gdf=joined_gdf,
                grid_value=current_fold_field,
                long_name=current_long_name,
                units=current_units,
                source=source,
                time=time,
                cell_size=cell_size,
                variable_name=filter_var,
                value_per_area=value_per_area,
                zero_is_value=zero_is_value
            )

            # Print or process verbose information
            if verbose:
                global_summary_stats = utils.dataframe_stats_line(dataframe=filtered_gdf, fold_field=fold_field, fold_function=fold_function)
                print(f"Global stats of {filter_var} before gridding : {global_summary_stats:.2f} km.")
                var_name = utils.replace_special_characters(filter_var)
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds_var, variable_name=var_name, value_per_area=value_per_area, cell_size=cell_size) * 1e-3
                print(f"Global stats of {filter_var} after gridding: {global_gridded_stats:.2f} km.")

            print("\n")
            dataset_list.append(ds_var)
        
        # Merge all datasets from different filtered GeoDataFrames
        ds = xr.merge(dataset_list)
        
    else:
        joined_gdf = utils.line_intersect(polygons_gdf, lines_gdf, fold_field=fold_field, fold_function=fold_function)

        # Determine fold_field, long_name, and units
        fold_field = "length_m"
        long_name = utils.determine_long_name_line(long_name, fold_field, variable_name)
        units = utils.determine_units_line(units, value_per_area)
        ds = utils.gridded_poly_2_xarray(
            polygon_gdf=joined_gdf,
            grid_value=fold_field,
            long_name=long_name,
            units=units,
            source=source,
            time=time,
            cell_size=cell_size,
            variable_name=variable_name,
            value_per_area=value_per_area,
            zero_is_value=zero_is_value
        )
        
        if verbose:
            global_summary_stats = utils.dataframe_stats_line(dataframe=lines_gdf, fold_field=fold_field, fold_function=fold_function)
            print(f"Global stats before gridding : {global_summary_stats:.2f} km.")
            global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, fold_field=fold_field, value_per_area=value_per_area, cell_size=cell_size) * 1e-3
            print(f"Global stats after gridding: {global_gridded_stats:.2f} km.")
    
    # save the xarray dataset
    if output_directory:
        if shapefile_path:
            base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    return ds



def point_2_grid(points_gdf=None, variable_name=None, long_name=None, units="value/grid-cell", source=None, time=None, 
                 cell_size=1, fold_field=None, fold_function="sum", attr_field=None, shapefile_path=None, 
                 output_directory=None, output_filename=None, value_per_area=False, zero_is_value=False, verbose=False):
    
    if points_gdf is None and shapefile_path is None:
        raise ValueError("Either 'points geodataframe' or 'shapefile directory' must be provided.")
    elif points_gdf is not None and shapefile_path is not None:
        raise ValueError("Only one of 'points geodataframe' or 'shapefile directory' should be provided.")

    # create gridded polygon
    polygons_gdf = create.create_gridded_polygon(cell_size=cell_size, out_polygon_path=None, grid_area=False)
    
    # spatial join
    if shapefile_path:
        points_gdf = gpd.read_file(shapefile_path)
    
    if attr_field is not None:
        unique_rows = points_gdf[attr_field].unique().tolist()
        dataset_list = []
        
        for filter_var in unique_rows:
            # Filter the GeoDataFrame
            filtered_gdf = points_gdf[points_gdf[attr_field] == filter_var].copy()
            joined_gdf = utils.point_spatial_join(polygons_gdf, filtered_gdf, fold_field=fold_field, fold_function=fold_function)

            # Determine fold_field, long_name, and units for the current iteration
            current_fold_field = fold_field or "count"
            current_long_name = utils.reverse_replace_special_characters(filter_var)
            current_units = utils.determine_units_point(units, value_per_area)

            # Convert joined GeoDataFrame to xarray dataset
            ds_var = utils.gridded_poly_2_xarray(
                polygon_gdf=joined_gdf,
                grid_value=current_fold_field,
                long_name=current_long_name,
                units=current_units,
                source=source,
                time=time,
                cell_size=cell_size,
                variable_name=filter_var,
                value_per_area=value_per_area,
                zero_is_value=zero_is_value
            )

            # Print or process verbose information
            if verbose:
                global_summary_stats = utils.dataframe_stats_point(dataframe=filtered_gdf, fold_field=current_fold_field, fold_function=fold_function)
                print(f"Global stats of {filter_var} before gridding : {global_summary_stats:.2f}")
                var_name = utils.replace_special_characters(filter_var)
                global_gridded_stats = utils.xarray_dataset_stats(dataset=ds_var, variable_name=var_name, value_per_area=value_per_area, cell_size=cell_size)
                print(f"Global stats of {filter_var} after gridding: {global_gridded_stats:.2f}")

            print("\n")
            dataset_list.append(ds_var)
        
        # Merge all datasets from different filtered GeoDataFrames
        ds = xr.merge(dataset_list)
        
    else:
        joined_gdf = utils.point_spatial_join(polygons_gdf, points_gdf, fold_field=fold_field, fold_function=fold_function)

        # Determine fold_field, long_name, and units
        fold_field = fold_field or "count"
        long_name = utils.determine_long_name_point(fold_field, variable_name, long_name, fold_function)
        units = utils.determine_units_point(units, value_per_area)
        
        ds = utils.gridded_poly_2_xarray(
            polygon_gdf=joined_gdf,
            grid_value=fold_field,
            long_name=long_name,
            units=units,
            source=source,
            time=time,
            cell_size=cell_size,
            variable_name=variable_name,
            value_per_area=value_per_area,
            zero_is_value=zero_is_value
        )

        if verbose:
            global_summary_stats = utils.dataframe_stats_point(dataframe=points_gdf, fold_field=fold_field, fold_function=fold_function)
            print(f"Global stats before gridding : {global_summary_stats:.2f}")
            global_gridded_stats = utils.xarray_dataset_stats(dataset=ds, variable_name=variable_name, value_per_area=value_per_area, cell_size=cell_size)
            print(f"Global stats after gridding: {global_gridded_stats:.2f}")
    
    # save the xarray dataset
    if output_directory:
        if shapefile_path:
            base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]
        utils.save_to_nc(ds, output_directory=output_directory, output_filename=output_filename, base_filename=base_filename)
    return ds


def plot_histogram(variable, dataset=None, bin_size=30, color='blue', plot_title=None, x_label=None, remove_outliers=False, log_transform=None, output_dir=None, filename=None, netcdf_directory=None):
    """
    Create a histogram for an array variable in an xarray dataset.
    Optionally remove outliers and apply log transformations.
    
    Parameters:
    - dataset: xarray.Dataset, the dataset containing the variables.
    - variable: str, the name of the variable to plot.
    - bin_size: int, optional, the number of bins in the histogram.
    - color: str, optional, the color of the histogram bars.
    - remove_outliers: bool, optional, whether to remove outliers.
    - log_transform: str, optional, the type of log transformation ('log10', 'log', 'log2').
    - netcdf_directory: str, optional: directory where netcdf file is located.
    
    Returns:
    - None, displays the plot.
    """
    plot.plot_histogram(variable, dataset, bin_size, color, plot_title, x_label, remove_outliers, log_transform, output_dir, filename, netcdf_directory)
    
def plot_scatter(variable1, variable2, dataset=None, dataset2=None, color='blue', x_label=None, y_label=None, plot_title=None, remove_outliers=False, log_transform_1=None, log_transform_2=None, equation=False, output_dir=None, filename=None, netcdf_directory=None, netcdf_directory2=None):
    """
    Create a scatter plot for two variables in an xarray dataset.
    Optionally remove outliers and apply log transformations.
    
    Parameters:
    - dataset: xarray.Dataset, the dataset containing the variables for the x-axis.
    - variable1: str, the name of the first variable to plot on the x-axis.
    - variable2: str, the name of the second variable to plot on the y-axis. If dataset2 is provided, this variable is from dataset2.
    - dataset2: xarray.Dataset, optional, a second dataset containing the variable for the y-axis.
    - remove_outliers: bool, optional, whether to remove outliers.
    - log_transform: str, optional, the type of log transformation ('log10', 'log', 'log2').
    - color: str, optional, the color of the scatter plot points.
    - netcdf_directory: str, optional: directory where netcdf file of the variable is located.
    - netcdf_directory2: str, optional: directory where netcdf file of the 2nd variable is located.
    
    Returns:
    - None, displays the plot.
    """
    plot.plot_scatter(variable1, variable2, dataset, dataset2, color, x_label, y_label, plot_title, remove_outliers, log_transform_1, log_transform_2, equation, output_dir, filename, netcdf_directory, netcdf_directory2)
    

def plot_time_series(variable, dataset=None, fold_function='sum', plot_type='both', color='blue', plot_label='Area Plot', x_label='Year', y_label='Value', plot_title='Time Series Plot', smoothing_window=None, output_dir=None, filename=None, netcdf_directory=None):
    """
    Create a line plot and/or area plot for a time series data variable.
    
    Parameters:
    - ds: xarray.Dataset, the dataset containing the variable to plot.
    - variable: str, the name of the variable to plot.
    - fold_function: str, the operation to apply ('sum', 'mean', 'max', 'std').
    - smoothing_window: int, optional, the window size for rolling mean smoothing.
    - plot_type: str, optional, the type of plot ('line', 'area', 'both'). Default is 'both'.
    - color: str, optional, the color of the plot. Default is 'blue'.
    - plot_label: str, optional, the label for the plot. Default is 'Area Plot'.
    - x_label: str, optional, the label for the x-axis. Default is 'Year'.
    - y_label: str, optional, the label for the y-axis. Default is 'Value'.
    - plot_title: str, optional, the title of the plot. Default is 'Time Series Plot'.
    - output_dir: str, optional, the directory to save the plot.
    - filename: str, optional, the filename to save the plot.
    - netcdf_directory: str, optional: directory where netcdf file is located.
    
    Returns:
    - None, displays the plot.
    """
    
    plot.plot_time_series(variable, dataset, fold_function, plot_type, color, plot_label, x_label, y_label, plot_title, smoothing_window, output_dir, filename, netcdf_directory)


def plot_hexbin(variable1, variable2, dataset=None, dataset2=None, color='pink_r', grid_size=30, x_label=None, y_label=None, plot_title=None, remove_outliers=False, log_transform_1=None, log_transform_2=None, output_dir=None, filename=None, netcdf_directory=None, netcdf_directory2=None):
    
    """
    Create a hexbin plot for two variables in an xarray dataset.

    Parameters:
    - dataset: xarray.Dataset, the dataset containing the variables for the x-axis.
    - variable1: str, the name of the first variable to plot on the x-axis.
    - variable2: str, the name of the second variable to plot on the y-axis. If dataset2 is provided, this variable is from dataset2.
    - dataset2: xarray.Dataset, optional, a second dataset containing the variable for the y-axis.
    - color: str, optional, the color map of the hexbin plot.
    - grid_size: int, optional, the number of hexagons in the x-direction.
    - x_label: str, optional, the label for the x-axis.
    - y_label: str, optional, the label for the y-axis.
    - plot_title: str, optional, the title for the plot.
    - remove_outliers: bool, optional, whether to remove outliers from the data.
    - log_transform_1: str, optional, the type of log transformation for variable1 ('log10', 'log', 'log2').
    - log_transform_2: str, optional, the type of log transformation for variable2 ('log10', 'log', 'log2').
    - netcdf_directory: str, optional: directory where netcdf file of the variable is located.
    - netcdf_directory2: str, optional: directory where netcdf file of the 2nd variable is located.

    Returns:
    - None, displays the plot.
    """
    
    plot.plot_hexbin(variable1, variable2, dataset, dataset2, color, grid_size, x_label, y_label, plot_title, remove_outliers, log_transform_1, log_transform_2, output_dir, filename, netcdf_directory, netcdf_directory2)
    

def plot_map(variable, dataset=None, color='hot_r', title='', label='', color_min=None, color_max=None, levels=10, output_dir=None, filename=None, netcdf_directory=None):
    
    """
    TODO: Need to write the docstring
    """
    plot.plot_map(variable, dataset, cmap_name=color, title=title, label=label, color_min=color_min, color_max=color_max, levels=levels, output_dir=output_dir, filename=filename, netcdf_directory=netcdf_directory)
    
    
def sum_variables(variables=None, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):

    """
    Sum specified variables in the xarray dataset. If no variables are specified, sum all variables
    except those starting with 'grid_area'. Fill NaNs with zero before summing, and convert resulting
    zeros back to NaNs.
    
    Parameters:
    - variables: list of str, the names of the variables to sum. If None, sum all variables except those
                 starting with 'grid_area'.
    - dataset: xarray.Dataset, optional, the dataset containing the variables.
    - new_variable_name: str, optional, the name of the new variable to store the sum.
    - time: optional, a specific time slice to select from the dataset.
    - netcdf_directory: str, optional: directory where netcdf file is located. 
    
    Returns:
    - xarray.Dataset, with the summed variable.
    """
    
    ds = calculate.sum_variables(variables, dataset, new_variable_name, time, netcdf_directory)
    return ds
    
def subtract_variables(variable1, variable2, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    
    """
    Subtract one variable from another in the xarray dataset.
    Fill NaNs with zero before subtracting, and convert resulting zeros back to NaNs.
    
    Parameters:
    - variable1: str, the name of the variable to subtract from.
    - variable2: str, the name of the variable to subtract.
    - dataset: xarray.Dataset, optional, the dataset containing the variables.
    - new_variable_name: str, optional, the name of the new variable to store the result.
    - time: optional, a specific time slice to select from the dataset.
    - netcdf_directory: str, optional: directory where netcdf file is located. 
    
    Returns:
    - xarray.Dataset, with the resulting variable.
    """
    ds = calculate.subtract_variables(variable1, variable2, dataset, new_variable_name, time, netcdf_directory)
    return ds
    


def divide_variables(variable1, variable2, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    """
    Divide one variable by another in the xarray dataset.
    Fill NaNs with zero before dividing, and convert resulting zeros back to NaNs.
    
    Parameters:
    - variable1: str, the name of the variable to be divided (numerator).
    - variable2: str, the name of the variable to divide by (denominator).
    - dataset: xarray.Dataset, optional, the dataset containing the variables.
    - new_variable_name: str, optional, the name of the new variable to store the result.
    - time: optional, a specific time slice to select from the dataset.
    - netcdf_directory: str, optional: directory where netcdf file is located. 
    
    Returns:
    - xarray.Dataset, with the resulting variable.
    """
    ds = calculate.divide_variables(variable1, variable2, dataset, new_variable_name, time, netcdf_directory)
    return ds
    

def multiply_variables(variables=None, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    """
    Multiply specified variables in the xarray dataset. If no variables are specified, multiply all variables.
    Fill NaNs with one before multiplying, and convert resulting ones back to NaNs.
    
    Parameters:
    - variables: list of str, the names of the variables to multiply. If None, multiply all variables.
    - dataset: xarray.Dataset, optional, the dataset containing the variables.
    - new_variable_name: str, optional, the name of the new variable to store the product.
    - time: optional, a specific time slice to select from the dataset.
    - netcdf_directory: str, optional: directory where netcdf file is located. 
    
    Returns:
    - xarray.Dataset, with the resulting variable.
    """
    
    ds = calculate.multiply_variables(variables, dataset, new_variable_name, time, netcdf_directory)
    
    return ds
    
def average_variables(variables=None, dataset=None, new_variable_name=None, time=None, netcdf_directory=None):
    """
    Average specified variables in the xarray dataset. If no variables are specified, average all variables
    except those starting with 'grid_area'. Fill NaNs with zero before averaging, and convert resulting
    zeros back to NaNs.
    
    Parameters:
    - variables: list of str, the names of the variables to average. If None, average all variables except those
                 starting with 'grid_area'.
    - dataset: xarray.Dataset, optional, the dataset containing the variables.
    - new_variable_name: str, optional, the name of the new variable to store the average.
    - time: optional, a specific time slice to select from the dataset.
    - netcdf_directory: str, optional: directory where netcdf file is located.
    
    Returns:
    - xarray.Dataset, with the averaged variable.
    """
    ds = calculate.average_variables(variables, dataset, new_variable_name, time, netcdf_directory)
    return ds


def grid_2_table(input_netcdf_path=None, ds=None, variable=None, time=None, grid_area=None, cell_size=1, aggregation=None, method='sum', verbose=False):
    """
    Process gridded data from an xarray Dataset to generate tabular data for different jurisdictions.

    Parameters:
    -----------
    input_netcdf_path : str, optional
        Netcdf path containing path location. 
    ds : xarray Dataset, optional
        Gridded dataset containing spatial information.
    variable : str, optional
        Variable name to be processed. If None, all variables in the dataset (excluding predefined ones) will be considered.
    time : str, optional
        Time slice for data processing. If provided, the nearest time slice is selected. If None, a default time slice is used.
    grid_area : str, optional
        Indicator to consider grid area during processing. If 'YES', the variable is multiplied by grid area.
    aggregation : str, optional
        Aggregation level for tabular data. If 'continent', the data will be aggregated at the continent level.
    method : str, optional
        Aggregation method. Options: 'sum', 'mean', 'max'.

    Returns:
    --------
    merged_df : pandas DataFrame
        Tabular data for different jurisdictions, including ISO3 codes, variable values, and optional 'Year' column.
    """
   
    df = utils.grid_2_table(input_netcdf_path=input_netcdf_path, ds=ds, variable=variable, time=time, 
                           grid_area=grid_area, cell_size=cell_size, aggregation=aggregation, method=method, 
                           verbose=verbose)
    return df


def get_netcdf_info(netcdf_path, variable_name=None):
    """
    Extract information about variables and dimensions from a NetCDF dataset.

    Parameters
    ----------
    netcdf_path : str
        The file path to the NetCDF dataset.
    variable_name : str, optional
        The prefix or complete name of the variable to filter. If not provided, all variables are included.

    Returns
    -------
    tuple
        A tuple containing lists of dimensions, short names, long names, units, & time values (if 'time' exists).
    """

    netcdf_info = get.get_netcdf_info(netcdf_path=netcdf_path, variable_name=variable_name)
    return netcdf_info


def country_2_iso3(df, column):
    """
    Convert country names in a DataFrame column to their corresponding ISO3 country codes.

    This function reads a JSON file containing country names and their corresponding ISO3 codes, then 
    maps the values from the specified column in the DataFrame to their ISO3 codes based on the JSON data. 
    The resulting ISO3 codes are added as a new column named 'ISO3'.

    Args:
        df (pandas.DataFrame): The DataFrame containing a column with country names.
        column (str): The name of the column in the DataFrame that contains country names.

    Returns:
        pandas.DataFrame: The original DataFrame with an additional 'ISO3' column containing the ISO3 country codes.

    Raises:
        FileNotFoundError: If the JSON file containing country mappings cannot be found.
        KeyError: If the specified column is not present in the DataFrame.
    """

    # Convert country names to ISO3
    base_directory = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_directory, "Names.json")
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


