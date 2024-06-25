
#TODO: add comments describing each attribute and method

class SesameToolboxInterface:
	def get_netcdf_info(self, netcdf_path, variable_name=None):
		pass

	def point_2_grid(self, shape_file_path, short_name, long_name, units="value/grid-cell", source=None, cell_size=1.0,
					 filename=None, time=None, zero_is_value=None, fold_field=None, fold_function="SUM",
					 attr_field=None, output_directory=None, output_filename=None):
		pass

	def line_2_grid(self, shape_file_path, short_name, long_name, units="km/grid-cell", source=None, cell_size=1.0,
					filename=None, time=None, zero_is_value=None, fold_field=None, fold_function="SUM", attr_field=None,
					output_directory=None, output_filename=None):
		pass

	def poly_2_grid(self, shape_file_path, short_name, long_name, units="km2/grid-cell", source=None, cell_size=1.0,
					filename=None, time=None, zero_is_value=None, attr_field=None, fraction=False, output_directory=None,
					output_filename=None):
		pass

	def table_2_grid(self, netcdf_variable, tabular_column, netcdf_file_path=None, csv_file_path=None, input_ds=None,
					 input_df=None, short_name=None, long_name=None, units="value/grid-cell", source=None, time=None,
					 filename=None, zero_is_value=None, value_per_sqm=None, verbose=False):
		pass

	def grid_2_grid(self, raster_path, fold_function, short_name, long_name, units="value/grid-cell", source=None,
					time=None, cell_size=1.0, zero_is_value=None, netcdf_variable=None, verbose=False,
					value_per_sqm=None, output_directory=None, output_filename=None):
		pass

	def grid_2_table(self, input_netcdf_path=None, ds=None, variable=None, time=None, grid_area=None, cell_size=1.0,
					 aggregation=None, fold_function='SUM', verbose=False, output_directory=None, output_filename=None):
		pass
