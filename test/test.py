import sesame as st
import os

# Generate line netcdf file
shapefile_path = os.path.join("data", "Global_Railways_WFP.shp")

variable_name = 'railway_length'
long_name = 'Total Railway Length in km'
source = 'Global Railways (WFP-World Food Programme SDI-T - Logistics Database)'
output_directory = os.path.join("data/")
output_filename = "line_2_grid"
ds = st.line_2_grid(lines_gdf=None, variable_name=variable_name, long_name=long_name, units="meter/grid-cell", source=source, time=None, 
                 cell_size=1, fold_field=None, fold_function="sum", attr_field=None, shapefile_path=shapefile_path, 
                 output_directory=output_directory, output_filename=output_filename, value_per_area=None, zero_is_value=False, verbose="yes")
print(ds)
