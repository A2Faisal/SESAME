from sesame_toolbox_interface import SesameToolboxInterface
import json
import os
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import sys
import utils
import calculate
import get
import create
import process_data
import plot

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')

class SesameConfig():
	def __init__(self):
		self.global_attr = {
	'Project': 'Surface Earth System Analysis and Modeling Environment (SESAME)',
	'Research Group': 'Integrated Earth System Dynamics',
	'Institution': 'McGill University',
	'Contact': 'eric.galbraith@mcgill.ca',
	'Data Version': 'V1.0'}
	
		# Load the config JSON file
		with open(DEFAULT_CONFIG_FILE, 'r') as f:
			self.config = json.load(f)
		
		self.arcpy = self.import_arcpy()

	def import_arcpy(self):
		"""
		Imports the `arcpy` module and sets up the environment for spatial analysis.

		Returns:
			module: The `arcpy` module with the environment configured.
		"""
		import arcpy
		from arcpy import env, SpatialReference, sa

		# overwriting the existing data
		arcpy.env.overwriteOutput = True

		# Defining the default coordinate system
		arcpy.CheckOutExtension("Spatial")
		arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("GCS_WGS_1984")

		return arcpy

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
			A tuple containing lists of dimensions, short names, long names, units, & time values (if 'time' exists)
			 for each variable within the dataset
		"""
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

#TODO
"""
field_name --> fold_field
unique_field_column --> attr_field
add bool var for number count 
count_points = default False, True if fold_field is None
"""
def point_2_grid(shapefile_path, short_name=None, long_name=None, units="value/grid-cell", source=None, cell_size=1,
				 time=None, fold_field=None, fold_function="SUM", attr_field=None, output_directory=None,
				 output_filename=None, value_per_sqm=False, verbose=False, config=SesameConfig()):
	"""
		Convert point-based shapefile data to a gridded raster dataset, summarizing values within grid cells.

		Parameters:
		-----------
		shapefile_path : str
			Path to the input point-based shapefile containing data to be converted.
		short_name : str
			Name of the variable.
		long_name : str
			A long name for the variable.
		units : str, optional
			Units of the variable, if available. Default is 'value/grid-cell'.
		source : str, optional
			Source information, if available. Default is None.
		cell_size : float, optional
			Size of the grid cells. Default is 1.0.
		time : str, optional
			Time slice for data processing. If provided, the nearest time slice is selected. If None, a default time
			 slice is used.
		zero_is_value: str, optional
			If the value is “yes”, then the function will treat zero as an existent value and 0 values will be
			considered while calculating mean and STD.
		fold_field : str, optional
			Field name in the input shapefile to be summarized within grid cells. If None, point counts will be
			summarized.
		fold_function : str, optional
			Summary statistic to apply to the field (e.g., 'SUM', 'MEAN'). Required if fold_field is specified.
		attr_field : str, optional
			Column containing unique values to be used for filtering and processing individual subsets.
		output directory: str, optional
			path to desired netcdf file location.
		output_filename: str, optional
			By default, it will save the netcdf file as the input filename by adding .nc as extension.
			Otherwise, a user can specify the file name.
		verbose: bool, optional
			if yes is mentioned then global statistics of before and after conversion will be printed.
		config=SesameConfig(): object, optional
		Contains arcpy input and supporting libraries and path of configuration files. 
        
		Returns:
		--------
		ds : xarray.Dataset
			Gridded raster dataset containing summarized values within grid cells.
		"""
	# Create a temporary folder
	temp_path = create.create_temp_folder(shapefile_path, "temp")

	# Create gridded polygon
	cell_size_name = process_data.replace_special_characters(str(cell_size))
	world_shp = create.create_gridded_polygon(config, temp_path, cell_size, verbose=verbose)
	world_shp = temp_path + "World_" + cell_size_name + "deg.shp"
	gdb_path = get.get_geodatabase(config, temp_path)

	# Get the base file name
	base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]
	# Check if there is any points that fall exactly on the boundary of gridded polygon
	selected, output_layer_names, num_records = config.arcpy.management.SelectLayerByLocation(in_layer=[shapefile_path],
																			 overlap_type="BOUNDARY_TOUCHES",
																			 select_features=world_shp,
																			 search_distance="",
																			 selection_type="NEW_SELECTION",
																			 invert_spatial_relationship="NOT_INVERT")
	
	if int(num_records) > 0:
		shapefile_path = process_data.adjust_shapefile_points(config, shapefile_path, world_shp, temp_path)

	if verbose:
		# Summary Stats
		if fold_field is not None:
			value = calculate.calculate_statistics(config, shapefile_path, fold_field)
			print(f"Global sum of {fold_field} column before gridding : {value:.2f}")
		else:
			count = int(config.arcpy.GetCount_management(shapefile_path).getOutput(0))
			print(f"Total number of points in the shapefile before gridding : {count}")

	if attr_field is not None:
		unique_values_list = get.get_unique_values(config, shapefile_path, attr_field)
		dataset_list = get.get_unique_point_values(config, unique_values_list, shapefile_path, gdb_path, temp_path,
												   fold_field, fold_function, attr_field, units, source, time,
												   cell_size, value_per_sqm)
		
		ds = utils.merge_ds_list(config, dataset_list)
	else:
		raster_path = utils.point_2_tif(config, shapefile_path, temp_path, gdb_path, cell_size, output_filename,
										fold_field, fold_function)

		if short_name is None or long_name is None:
			short_name = base_filename
			long_name = process_data.reverse_replace_special_characters(short_name)

		# Convert the regridded raster to a dataset
		ds = utils.regridded_tif_2_ds(config, raster_path, short_name, long_name, units, source, time, cell_size)
		if verbose:
			# Print the global sum
			global_sum = calculate.calculate_statistics_from_dataset(ds, short_name)
			print(f"Global raster sum after gridding : {global_sum:.2f}")

	utils.save_to_nc(output_directory, output_filename, time, ds, base_filename)
	print("The point file has been converted to gridded dataset.")
	return ds

def line_2_grid(shapefile_path, short_name, long_name, units="km/grid-cell", source=None, cell_size=1.0,
				time=None, fold_field=None, fold_function="SUM", attr_field=None,
				output_directory=None, output_filename=None, config=SesameConfig()):
	"""
		Convert line-based shapefile data to a gridded dataset, summarizing length or a specified field within polygons.

		Parameters:
		-----------
		shapefile_path : str
			Path to the input line-based shapefile containing data to be converted.
		short_name : str
			Name of the variable.
		long_name : str
			A long name for the variable.
		units : str
			Units of the variable, if available. Default is 'km/grid-cell'.
		source : str, optional
			Source information, if available. Default is None.
		cell_size : float, optional
			Size of the grid cells. Default is 1.0.
		time : object, optional
			Time information associated with the dataset.
		fold_field : str, optional
			Field name in the input shapefile to be summarized within polygons. If None, line lengths will be
			summarized.
		fold_function : str, optional
			Summary statistic to apply to the field (e.g., 'SUM', 'MEAN'). Required if fold_field is specified.
		attr_field : str, optional
			Column in the shapefile representing unique values for grouping.
		output directory: str, optional
			path to desired netcdf file location.
		output_filename: str, optional
			By default, it will save the netcdf file as the input filename by adding .nc as extension.
			Otherwise, a user can specify the file name.
		config=SesameConfig(): object, optional
			Contains arcpy input and supporting libraries and path of configuration files. 
        
		Returns:
		--------
		xr.Dataset
			Gridded dataset containing summarized information based on the specified parameters.
	"""
	# Creating temporary folder
	temp_path = create.create_temp_folder(shapefile_path, "temp")
	# check coordinate system of the shapefile
	shapefile_path = process_data.check_shapefile_projection(config, shapefile_path, temp_path)
	gdb_path = get.get_geodatabase(config, temp_path)

	# Get the base file name
	base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]
	line_length = calculate.calculate_length(config, shapefile_path)
	print(f"Total length of lines in the shapefile: {line_length:.2f} km.")

	if attr_field is not None:
		# Get unique values in the specified column
		unique_values_list = get.get_unique_values(config, shapefile_path, attr_field)
		dataset_list = get.get_unique_line_values(config, unique_values_list, shapefile_path, gdb_path, temp_path,
												  fold_field, fold_function, attr_field, units, source, time,
												  cell_size)
		# Merge the list of datasets into one
		ds = utils.merge_ds_list(config, dataset_list)
	else:
		# Convert line-based shapefile to raster
		raster_path = utils.line_2_tif(config, shapefile_path, temp_path, gdb_path, cell_size, output_filename, fold_field,
									   fold_function)
		if short_name is None or long_name is None:
			short_name = base_filename
			long_name = process_data.reverse_replace_special_characters(short_name)

		ds = utils.regridded_tif_2_ds(config, raster_path, short_name, long_name, units, source, time, cell_size)

		# Print the global sum
		global_sum = ds[short_name].sum().values
		print(f"Global raster sum after gridding : {global_sum:.2f} km.")

	utils.save_to_nc(output_directory, output_filename, time, ds, base_filename)
	print("The line file has been converted to gridded dataset.")
	return ds

# TODO:  change fraction to bool instead of string
def poly_2_grid(shapefile_path, short_name=None, long_name=None, units="km2/grid-cell", source=None, cell_size=1.0,
				time=None, zero_is_value=None, attr_field=None, fraction=False, output_directory=None,
				output_filename=None, config=SesameConfig()):
	"""
	Convert polygon-based shapefile data to a Geodatabase raster format, summarizing values within grid cells.

	Parameters:
	-----------
	shapefile_path : str
		Path to the input polygon-based shapefile containing data to be converted.
	short_name : str
		Name of the variable.
	long_name : str
		A long name for the variable.
	units : str, optional
		Units of the variable, if available. Default is 'km2/grid-cell'.
	source : str, optional
		Source information, if available. Default is None.
	cell_size : float, optional
		Size of raster cells. Default is 1.0.
	time : str, optional
		Time information for the dataset.
	zero_is_value: str, optional
		If the value is “yes”, then the function will treat zero as an existent value and 0 values will be
		considered while calculating mean and STD.
	attr_field : str, optional
		Field name in the input shapefile used to filter unique values.
	fraction : bool, optional
		Whether to calculate fraction data. Options: "Yes" or None (default).
	output directory: str, optional
			path to desired netcdf file location.
	output_filename: str, optional
		By default, it will save the netcdf file as the input filename by adding .nc as extension.
		Otherwise, a user can specify the file name.
	config=SesameConfig(): object, optional
		Contains arcpy input and supporting libraries and path of configuration files. 

	Returns:
	--------
	ds : xarray.Dataset
		Resulting dataset after converting and summarizing the polygon-based shapefile data.
	"""
	# Creating temporary folder
	temp_path = create.create_temp_folder(shapefile_path, "temp")

	# Create gridded polygon
	world_shp = create.create_gridded_polygon(config, temp_path, cell_size, grid_area='yes')
	cell_size_name = process_data.replace_special_characters(str(cell_size))
	world_shp = temp_path + "World_" + cell_size_name + "deg.shp"
	gdb_path = get.get_geodatabase(config, temp_path)

	# Get the base file name
	base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]
	if attr_field is not None:
		unique_values_list = get.get_unique_values(config, shapefile_path, attr_field)
		dataset_list = get.get_unique_polygon_values(config, unique_values_list, shapefile_path, gdb_path, temp_path,
													 fraction, attr_field, units, source, time, cell_size,
													 zero_is_value)
		ds = utils.merge_ds_list(config, dataset_list)
	elif fraction is not None:
		#TODO: add assertion that there is only one field, if not tell user to specify the attr_column (unique field columns)
		value = get.get_poly_area(config, shapefile_path)
		# Print the results
		print(f"Global sum of polygon area before gridding : {value:.2f} km2.")

		raster_path = utils.poly_2_tif(config, shapefile_path, temp_path, gdb_path, cell_size, output_filename, fraction)
		if fraction is True:
			units = "fraction"

		if short_name is None or long_name is None:
			short_name = base_filename
			long_name = process_data.reverse_replace_special_characters(short_name)

		ds = utils.regridded_tif_2_ds(config, raster_path, short_name, long_name, units, source, time, cell_size, zero_is_value)
	else:
		# TODO: add assertion that there is only one field, if not tell user to specify the attr_column (unique field columns)

		value = get.get_poly_area(config, shapefile_path)
		print(f"Global sum of polygon area before gridding : {value:.2f} km2.")

		raster_path = utils.poly_2_tif(config, shapefile_path, temp_path, gdb_path, cell_size, output_filename, fraction)
		if short_name is None or long_name is None:
			short_name = base_filename
			long_name = process_data.reverse_replace_special_characters(short_name)

		ds = utils.regridded_tif_2_ds(config, raster_path, short_name, long_name, units, source, time, cell_size, zero_is_value)
		# Print the global sum
		global_sum = ds[short_name].sum().values
		print(f"Global raster sum after gridding : {global_sum:.2f} km2.")

	utils.save_to_nc(output_directory, output_filename, time, ds, base_filename)
	print("The polygon file has been converted to gridded dataset.")
	return ds


def table_2_grid(netcdf_variable, tabular_column, netcdf_file_path=None, csv_file_path=None, input_ds=None,
				 input_df=None, short_name=None, long_name=None, units="value/grid-cell", source=None,
				 filename=None, time=None, zero_is_value=None, value_per_sqm=None, verbose=False):
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
			a tabular file where data is stored based on their juriconfigiction or ISO3 code. The csv file must hold a
			column named “ISO3”. If not, then users must use juriconfigiction_2_ISO3 function to convert the country
			name to their corresponding ISO3 code.
		input_ds : xarray.Dataset
			Input NetCDF dataset with spatial coordinates.
		input_df : pandas.DataFrame
			Input tabular dataset containing values to be distributed spatially.
		short_name : str, optional
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
			juriconfigiction where surrogate variable is missing and tabular data is evenly distributed over the
			juriconfigiction, the ISO3 codes of evenly distributed countries will also be printed.

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

	if short_name is None:
		short_name = long_name if long_name is not None else tabular_column

	if long_name is None:
		long_name = short_name if short_name is not None else tabular_column

	# check the netcdf resolution
	cell_size = abs(float(input_ds['lat'].diff('lat').values[0]))
	cell_size_str = str(cell_size)
	country_ds = utils.load_country_fraction_dataset(config, cell_size_str)

	input_ds, country_ds, a = utils.adjust_datasets(input_ds, country_ds, time)
	print(f"Distributing {short_name} onto {netcdf_variable}.")

	new_ds = create.create_new_ds(config, input_ds, tabular_column, country_ds, netcdf_variable, input_df, verbose)

	for var_name in new_ds.data_vars:
		a += np.nan_to_num(new_ds[var_name].to_numpy())

	da = xr.DataArray(a, coords={'lat': input_ds['lat'], 'lon': input_ds['lon']}, dims=['lat', 'lon'])

	if verbose:
		print(f"Global sum of juriconfigictional dataset : {input_df[[tabular_column]].sum().item()}")
		print(f"Global sum of gridded dataset : {da.sum().item()}\n")

	if value_per_sqm and value_per_sqm.upper() == "YES":
		da = utils.normalize_by_area(config, cell_size_str, da)

		if units == 'value/grid-cell':
			units = 'value m-2'

		ds = utils.da_to_ds(config, da, short_name, long_name, units, source, time, cell_size, zero_is_value)
	else:
		ds = utils.da_to_ds(config, da, short_name, long_name, units, source, time, cell_size, zero_is_value)
	return ds

#TODO: change converstion_type to fold_function
def grid_2_grid(raster_path, fold_function, short_name, long_name, units="value/grid-cell", source=None,
				time=None, cell_size=1.0, zero_is_value=None, netcdf_variable=None, verbose=False,
				value_per_sqm=None, output_directory=None, output_filename=None, config=SesameConfig()):
	"""
	Convert raster data (TIFF or netCDF) to a re-gridded xarray dataset.

	Parameters
	----------
	raster_path : str
		File path to the input raster data. Should be either a TIFF or netCDF file.
	fold_function : str
		Type of conversion to perform ('SUM', 'MEAN', or 'MAX').
	short_name : str, optional
		Name of the variable. Default is None.
	long_name : str, optional
		A long name for the variable. Default is None.
	units : str, optional
		Units of the variable. Default is 'value/grid'.
	source : str, optional
		Source information, if available. Default is None.
	time : str, optional
		Time information for the dataset.
	cell_size : float, optional
		Size of raster cells. Default is 1.0.
	zero_is_value: str, optional
		If the value is “yes”, then the function will treat zero as an existent value and 0 values will be
		considered while calculating mean and STD.
	netcdf_variable : str, optional
		Variable in the netCDF data to be converted. Required if the input is a netCDF file.
	verbose: bool, optional. Default is False
		 If True, the global gridded sum of before and after the re-gridding operation will be printed.
	value_per_sqm: str, optional
		If input is “yes”, then the value will be transformed into “value m-2”.
	output directory: str, optional
			path to desired netcdf file location.
	output_filename: str, optional
		By default, it will save the netcdf file as the input filename by adding .nc as extension.
		Otherwise, a user can specify the file name.
	config=SesameConfig(): object, optional
		Contains arcpy input and supporting libraries and path of configuration files. 

	Returns
	-------
	xarray.Dataset
		Re-gridded xarray dataset containing the converted raster data.
	"""
	# creating temporary folder
	temp_path = create.create_temp_folder(raster_path, "temp")

	# Determine the file extension and base_filename
	file_extension = os.path.splitext(raster_path)[1]
	base_filename = os.path.splitext(os.path.basename(raster_path))[0]

	if file_extension == ".tif":
		print("Reading the tif file.")
		# Check and potentially modify the coordinate system of the raster
		raster_path = utils.regrid(config, raster_path, temp_path)
		# Convert TIFF data to a re-gridded dataset
		ds = utils.from_tif(config, raster_path, temp_path, short_name, long_name, units, source, cell_size, time,
							zero_is_value, fold_function, verbose, value_per_sqm)
	elif file_extension == ".nc" or file_extension == ".nc4":
		# Convert netCDF to TIFF
		print("Reading the nc file.")
		netcdf_tif_path = utils.to_tif(config, raster_path, netcdf_variable, temp_path, time)

		# Check and potentially modify the coordinate system of the raster projection
		projected_raster_path = utils.regrid(config, netcdf_tif_path, temp_path)
		# Convert netCDF data to a re-gridded dataset
		ds = utils.from_tif(config, projected_raster_path, temp_path, short_name, long_name, units, source, cell_size,
							time, zero_is_value, fold_function, verbose, value_per_sqm)
	else:
		# Print an error message for unrecognized file types
		print("Error: File type is not recognized. File type should be either TIFF or netCDF file.")
		return None

	utils.save_to_nc(output_directory, output_filename, time, ds, base_filename)
	return ds

def grid_2_table(input_netcdf_path=None, ds=None, variable=None, time=None, grid_area=None, cell_size=1.0,
				 aggregation=None, fold_function='SUM', verbose=False, output_directory=None, output_filename=None, config=SesameConfig()):
	"""
	Process gridded data from an xarray Dataset to generate tabular data for different juriconfigictions.

	Parameters:
	-----------
	input_netcdf_path : str, optional
		Netcdf path containing path location.
	ds : xarray Dataset, optional
		Gridded dataset containing spatial information.
	variable : str, optional
		Variable name to be processed. If None, all variables in the dataset (excluding predefined ones) will be
		considered.
	time : str, optional
		Time information for the dataset.
	grid_area : str, optional
		Indicator to consider grid area during processing. If 'YES', the variable is multiplied by grid area.
	cell_size : float, optional
		Size of raster cells. Default is 1.0.
	aggregation : str, optional
		Aggregation level for tabular data. If 'continent', the data will be aggregated at the continent level.
	fold_function : str, optional
		Aggregation fold_function. Options: 'sum', 'mean', 'max'.
	verbose: bool, optional
		if yes is mentioned then global statistics of before and after conversion will be printed.
	output directory: str, optional
			path to desired netcdf file location.
	output_filename: str, optional
		By default, it will save the netcdf file as the input filename by adding .nc as extension.
		Otherwise, a user can specify the file name.

	Returns:
	--------
	merged_df : pandas DataFrame
		Tabular data for different juriconfigictions, including ISO3 codes, variable values, and optional 'Year' column.
	"""
	if input_netcdf_path:
		ds = xr.load_dataset(input_netcdf_path)

	if not isinstance(ds, xr.Dataset):
		raise ValueError("Please provide either netcdf or xarray dataset.")

	# Check if a specific variable is provided, otherwise consider all variables in the dataset
	if variable is not None:
		variables_list = [variable]
	else:
		# Get the list of variables except the specified ones
		exclude_vars = ["time", "lat", "lon", "latb", "lonb", "grid_area_1deg", "grid_area_0_5deg"]
		variables_list = [var for var in ds.variables if var not in exclude_vars]
		print(f"List of variables in the dataset: {variables_list}")

	# Load ISO3 to continent mapping from CSV
	try:
		iso3_continent_df = pd.read_csv(config.config['file_paths']['ISO3_Country_Continent_UN'], encoding='utf-8')
	except UnicodeDecodeError:
		# Try a different encoding if 'utf-8' fails
		iso3_continent_df = pd.read_csv(config.config['file_paths']['ISO3_Country_Continent_UN'], encoding='latin1')

	# Initialize an empty list to store DataFrames
	dataframes = []
	# Loop through each variable in the dataset
	for var in variables_list:
		dataframes = create.create_df(config, dataframes, var, verbose, cell_size, grid_area, fold_function, time, ds)

	# If a specific time is provided, add a 'Year' column to the resulting DataFrame
	merged_df = utils.add_year_col(time, dataframes)

	fold_function_upper = fold_function.upper()
	# Merge DataFrames based on 'ISO3' column
	for df in dataframes[1:]:
		merged_df = pd.merge(merged_df, df, on='ISO3')

	if aggregation == 'region':
		continent_df = pd.merge(merged_df, iso3_continent_df[['ISO-alpha3 Code', 'Region SESAME']], left_on='ISO3',
								right_on='ISO-alpha3 Code')
		if fold_function_upper in ["SUM", "MEAN", "MAX", "STD"]:
			agg_func = fold_function_upper.lower()  # Method names in DataFrame are lowercase
			merged_df = continent_df.groupby('Region SESAME').agg({var: agg_func}).reset_index()
		else:
			sys.exit(1)

	if verbose:
		print(f"Global tabular {fold_function_upper}: {getattr(merged_df[var], fold_function)()}")

	if output_filename:
		merged_df.to_csv(output_directory + "output_table.csv")

	return merged_df


def table_2_point(csv_path=None, dataframe=None, lat_column=None, lon_column=None, output_directory=None, output_filename=None):
    """
    Convert a table with latitude and longitude columns to a GeoDataFrame with point geometries and optionally save it as a shapefile.

    Parameters:
    -----------
    csv_path : str, optional
        Path to the CSV file containing the table data. Either `csv_path` or `dataframe` must be provided.
    dataframe : pd.DataFrame, optional
        DataFrame containing the table data. Either `csv_path` or `dataframe` must be provided.
    lat_column : str, optional
        Name of the column containing latitude values. Required if `csv_path` or `dataframe` is provided.
    lon_column : str, optional
        Name of the column containing longitude values. Required if `csv_path` or `dataframe` is provided.
    output_directory : str, optional
        Directory where the output shapefile will be saved. Required if saving the GeoDataFrame to a file.
    output_filename : str, optional
        Filename for the output shapefile, without the extension. Required if saving the GeoDataFrame to a file.

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame with point geometries created from the latitude and longitude columns.
    str, optional
        The path to the saved shapefile if `output_directory` and `output_filename` are provided.

    Raises:
    -------
    ValueError
        If neither `csv_path` nor `dataframe` is provided, or if `lat_column` and `lon_column` are not provided.
    """
    
    if csv_path is None and dataframe is None:
        raise ValueError("Either 'csv_path' or 'dataframe' must be provided.")

    if csv_path:
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        df = dataframe
        
    if lat_column and lon_column:
        # Ensure column names match with your DataFrame
        geometry = [Point(lon, lat) for lat, lon in zip(df[lat_column], df[lon_column])]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    if output_directory and output_filename:
        # Save the GeoDataFrame to a shapefile
        output_shapefile = output_directory + output_filename + ".shp"
        gdf.to_file(output_shapefile)
        return output_shapefile
        
    return gdf


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
    

def plot_map(variable, dataset=None, cmap_name='hot_r', title='', label='', color_min=None, color_max=None, levels=10, output_dir=None, filename=None, netcdf_directory=None):
    
    """
    TODO: Need to write the docstring
    """
    plot.plot_map(variable, dataset, cmap_name, title, label, color_min, color_max, levels, output_dir, filename, netcdf_directory)
    
    
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


