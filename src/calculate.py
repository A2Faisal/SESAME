import os
import numpy as np
import utils
import xarray as xr
import pandas as pd
import sys



def calculate_length(sd, shape_file_path):
    output_file = shape_file_path

    # Calculate length for each line in the shapefile
    length = sd.arcpy.management.CalculateGeometryAttributes(output_file, [["Length", "LENGTH_GEODESIC"]],
                                                               "KILOMETERS", "",
                                                               coordinate_format="SAME_AS_INPUT")[0]

    line_length = calculate_statistics(sd, shape_file_path, "Length")
    return line_length


def calculate_raw_global_value(conversion_type, arr, zero_is_value):
    # Calculate raw global value based on the conversion type
    if conversion_type.upper() == 'SUM':
        raw_global_value = arr.sum()
    elif conversion_type.upper() == 'MEAN':
        # Handling zero values based on the 'zero_is_value' parameter
        if zero_is_value and zero_is_value.upper() == "YES":
            raw_global_value = arr.mean()
        else:
            mean_arr = arr.astype(np.float64)
            mean_arr[mean_arr == 0] = np.nan
            # Calculate the mean
            raw_global_value = np.nanmean(mean_arr)
    elif conversion_type.upper() == 'MAX':
        raw_global_value = arr.max()
    else:
        raise ValueError("Conversion should be either SUM, MEAN, or MAX.")
    return raw_global_value


def calculate_grid_resolution(resolution):
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


def calculate_statistics(sd, shapefile, column_to_analyze):
    # Calculate the total sum
    total_sum = 0

    # Use arcpy's SearchCursor to iterate through rows in the shapefile
    with sd.arcpy.da.SearchCursor(shapefile, [column_to_analyze]) as cursor:
        for row in cursor:
            # Add the value of the specified column to the total sum
            total_sum += row[0]
    return total_sum


def calculate_statistics_from_dataset(ds, variable_name):
    # Calculate the sum
    total_sum = ds[variable_name].sum().values
    return total_sum


def calculate_padded_array(num_lat, num_lon, arr):
    num_rows, num_cols = arr.shape

    # Check if the dimensions are perfectly divisible
    if num_rows % num_lat == 0 and num_cols % num_lon == 0:
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
        padded_arr = np.pad(arr, ((lat_padding_start, lat_padding_end), (lon_padding_start, lon_padding_end)),
                            mode='constant', constant_values=0)
    return padded_arr


def calculate_aligned_array(num_lat, num_lon, padded_arr):
    # Calculate factors for latitude and longitude
    lat_factor = padded_arr.shape[0] // num_lat
    lon_factor = padded_arr.shape[1] // num_lon

    # Check if the dimensions are divisible
    if padded_arr.shape[0] % lat_factor != 0 or padded_arr.shape[1] % lon_factor != 0:
        print("Warning: padded array's dimensions are not perfectly divisible!")

    aligned_arr = np.zeros((num_lat, num_lon, lat_factor, lon_factor))

    for i in range(num_lat):
        for j in range(num_lon):
            lat_start = i * lat_factor
            lat_end = (i + 1) * lat_factor
            lon_start = j * lon_factor
            lon_end = (j + 1) * lon_factor

            aligned_arr[i, j] = padded_arr[lat_start:lat_end, lon_start:lon_end]

    return aligned_arr


def calculate_regridded_global_value(conversion_type, num_lat, num_lon, aligned_arr, cell_size, zero_is_value):
    lat_resolution = cell_size
    lon_resolution = cell_size
    lat = np.linspace(90 - lat_resolution / 2, -90 + lat_resolution / 2, num=num_lat)
    lon = np.linspace(-180 + lon_resolution / 2, 180 - lon_resolution / 2, num=num_lon)

    # Create an xarray DataArray with dimensions and coordinates
    da = xr.DataArray(aligned_arr, dims=("lat", "lon", "lat_factor", "lon_factor"),
                      coords={"lat": lat, "lon": lon})
    # Perform the aggregation over the lat_factor and lon_factor dimensions
    if conversion_type.upper() == 'SUM':
        da_agg = da.sum(dim=['lat_factor', 'lon_factor'])
        regridded_global_value = da_agg.sum().item()
    elif conversion_type.upper() == 'MEAN':
        if zero_is_value and zero_is_value.upper() == "YES":
            da_agg = da.mean(dim=['lat_factor', 'lon_factor'])
            regridded_global_value = da_agg.mean().item()
        else:
            # Calculate the sum and count of non-zero values
            da_sum = da.sum(dim=['lat_factor', 'lon_factor'])
            da_count = da.where(da != 0).count(dim=['lat_factor', 'lon_factor'])
            # Calculate the mean, handling division by zero
            da_agg_nan = da_sum / da_count.where(da_count != 0, np.nan)
            # Replace nan values with 0
            da_agg = da_agg_nan.fillna(0)
            regridded_global_value = da_agg_nan.mean().item()
    elif conversion_type.upper() == 'MAX':
        da_agg = da.max(dim=['lat_factor', 'lon_factor'])
        regridded_global_value = da_agg.max().item()
    else:
        raise ValueError("Conversion should be either SUM, MEAN, or MAX")
    return regridded_global_value, da_agg


def calculate_ds(sd, value_per_sqm, da, short_name, long_name, units, source, cell_size, time, zero_is_value):
    # Get the directory of the current script
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')
    
    if value_per_sqm and value_per_sqm.upper() == "YES":
        cell_size_str = str(cell_size)
        if cell_size_str == "1" or cell_size_str == "1.0":
            grid_ds = xr.load_dataset(sd.config['file_paths']['grid_area_1deg'])
            da = da / grid_ds["grid_area_1deg"]
        elif cell_size_str == "0.5":
            grid_ds = xr.load_dataset(sd.config['file_paths']['grid_area_0_5deg'])
            da = da / grid_ds["grid_area_0_5deg"]
        elif cell_size_str == "0.25":
            grid_ds = xr.load_dataset(sd.config['file_paths']['grid_area_0_25deg'])
            da = da / grid_ds["grid_area_0_25deg"]

        if units is None:
            units = 'value/grid-cell'

        ds = utils.da_to_ds(sd, da, short_name, long_name, units, source, time, cell_size, zero_is_value)
    else:
        ds = utils.da_to_ds(sd, da, short_name, long_name, units, source, time, cell_size, zero_is_value)
    return ds


def calculate_fold_function(fold_function, ds_var):
    data = None

    if fold_function.upper() == "SUM":
        data = ds_var.sum()
    elif fold_function.upper() == "MEAN":
        data = ds_var.mean()
    elif fold_function.upper() == "MAX":
        data = ds_var.max()
    elif fold_function.upper() == "STD":
        data = ds_var.std()
    else:
        sys.exit(1)
    return data


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

