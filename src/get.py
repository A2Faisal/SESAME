import os
import utils
import calculate
import process_data
import xarray as xr

def get_geodatabase(sd, temp_path):
    # Create a geodatabase if it does not already exist
    gdb_path = temp_path + "default_gdb.gdb"

    if not sd.arcpy.Exists(gdb_path):
        sd.arcpy.CreateFileGDB_management(temp_path, "default_gdb")

    gdb_path = os.path.join(temp_path + "default_gdb.gdb", '')
    return gdb_path


def get_unique_point_values(sd, unique_values_list, shape_file_path, gdb_path, temp_path, field_name,
                            field_summary, unique_field_column, units, source, time, cell_size, 
                            zero_is_value, value_per_sqm=False):
    dataset_list = []
    for value in unique_values_list:
        # Create a feature layer from the shapefile
        sd.arcpy.MakeFeatureLayer_management(shape_file_path, "FeatureLayer")

        # Apply the Definition Query to filter the shapefile
        expression = str(unique_field_column + " = '" + value + "'")
        value_shapefile = sd.arcpy.SelectLayerByAttribute_management("FeatureLayer",
                                                                     "NEW_SELECTION", expression)

        # Use the GetCount function to get the number of selected features
        selected_count = int(sd.arcpy.GetCount_management(value_shapefile).getOutput(0))
        print(f"Number of points found for {value} : {selected_count}")

        # Summary Stats
        if field_name is not None:
            # Call the function to calculate statistics
            val = calculate.calculate_statistics(sd, value_shapefile, field_name)
            # Print the results
            print(f"Global sum of {value} before gridding : {val:.2f}")
        else:
            print(f"Number of {value} found before gridding : {selected_count}")

        # Replace white spaces with underscores
        filename = process_data.replace_special_characters(value)
        short_name = filename
        long_name = process_data.reverse_replace_special_characters(short_name)

        raster_path = utils.point_2_tif(sd, value_shapefile, temp_path, gdb_path, cell_size, short_name, field_name,
                                        field_summary)

        ds = utils.regridded_tif_2_ds(sd, raster_path, short_name, long_name, units, source, time, cell_size, zero_is_value)

        # Call the function to calculate statistics
        val = calculate.calculate_statistics_from_dataset(ds, filename)
        # Print the results
        print(f"Global raster sum of {value} after gridding : {val:.2f}")
        
        if value_per_sqm:
            if units == "value/grid-cell":
                units = "value m-2"
            da = ds.to_array()
            ds = calculate.calculate_ds(sd, value_per_sqm, da, short_name, long_name, units, source, cell_size, 
                                        time, zero_is_value)
        dataset_list.append(ds)
        print("------------------------------------------")
    return dataset_list


def get_unique_line_values(sd, unique_values_list, shape_file_path, gdb_path, temp_path, field_name,
                           field_summary, unique_field_column, units, source, time, cell_size, zero_is_value):
    dataset_list = []

    for value in unique_values_list:
        # Create a feature layer from the shapefile
        sd.arcpy.MakeFeatureLayer_management(shape_file_path, "FeatureLayer")

        # Apply the Definition Query to filter the shapefile
        expression = str(unique_field_column + " = '" + value + "'")
        value_shapefile = sd.arcpy.SelectLayerByAttribute_management("FeatureLayer", "NEW_SELECTION", expression)

        # Use the GetCount function to get the number of selected features
        selected_count = int(sd.arcpy.GetCount_management(value_shapefile).getOutput(0))
        print(f"Number of line found for {value} : {selected_count}")

        # Summary Stats
        line_length = calculate.calculate_statistics(sd, value_shapefile, "Length")
        print(f"Total length of {value} lines in the shapefile: {line_length:.2f} km.")

        # Replace white spaces and special values
        filename = process_data.replace_special_characters(value)
        raster_path = utils.line_2_tif(value_shapefile, temp_path, gdb_path, field_name, field_summary)

        # Update short_name and long_name with file-specific information
        short_name = filename
        long_name = process_data.reverse_replace_special_characters(short_name)

        # Convert raster to xarray Dataset
        ds = utils.regridded_tif_2_ds(sd, raster_path, short_name, long_name, units, source, time, cell_size,
                                      zero_is_value)

        # Call the function to calculate statistics
        val = calculate.calculate_statistics_from_dataset(sd, filename)
        print(f"Global raster sum of {value} after gridding : {val:.2f} km.")

        # Append the dataset to the list
        dataset_list.append(ds)
        print("------------------------------------------")
    return dataset_list


def get_unique_polygon_values(sd, unique_values_list, shape_file_path, gdb_path, temp_path, fraction,
                              unique_field_column, units, source, time, cell_size, zero_is_value, value_per_sqm=False):
    dataset_list = []

    for value in unique_values_list:
        # Create a feature layer from the shapefile
        sd.arcpy.MakeFeatureLayer_management(shape_file_path, "FeatureLayer")

        # Apply the Definition Query to filter the shapefile
        expression = str(unique_field_column + " = '" + value + "'")
        value_shapefile = sd.arcpy.SelectLayerByAttribute_management("FeatureLayer", "NEW_SELECTION", expression)

        # Use the GetCount function to get the number of selected features
        selected_count = int(sd.arcpy.GetCount_management(value_shapefile).getOutput(0))
        print(f"Number of polygon found for {value} : {selected_count}")

        # Summary Stats
        val = get_poly_area(sd, value_shapefile)  # Call the function to calculate statistics
        # Print the results
        print(f"Global sum of {value} before gridding : {val:.2f} km2.")

        # Replace whitespace, commas, hyphens, ampersansd.ds, and other special characters with underscores
        filename = process_data.replace_special_characters(value)

        raster_path = utils.poly_2_tif(sd, value_shapefile, temp_path, gdb_path, cell_size, filename, fraction)

        short_name = filename
        long_name = process_data.reverse_replace_special_characters(short_name)

        if fraction and fraction.upper() == "YES":
            units = "fraction"

        ds = utils.regridded_tif_2_ds(sd, raster_path, short_name, long_name, units, source, time, cell_size, 
                                      zero_is_value)

        # Call the function to calculate statistics
        if fraction is None:
            val = calculate.calculate_statistics_from_dataset(sd, filename)
            # Print the results
            print(f"Global raster sum of {value} after grdding : {val:.2f} km2")

        dataset_list.append(ds)
        print("------------------------------------------")
    return dataset_list

def get_unique_values(sd, input_shapefile, field_name):
    # Use a set to collect unique values
    unique_values = set()

    # Open a SearchCursor to iterate through the recorsd.ds
    with sd.arcpy.da.SearchCursor(input_shapefile, field_name) as cursor:
        for row in cursor:
            value = row[0]  # Assuming the field is the first (0-based) field in the attribute table
            unique_values.add(value)

    # Convert the set of unique values to a list for easier handling (if needed)
    unique_values_list = list(unique_values)
    print("Unique field values are: ", unique_values_list)

    return unique_values_list


def get_poly_area(sd, poly_shapefile):
    # Calculate geodesic area for each polygon in the shapefile
    poly_shapefile = sd.arcpy.management.CalculateGeometryAttributes(poly_shapefile,
                                                                       [["poly_area", "AREA_GEODESIC"]], "",
                                                                       "SQUARE_KILOMETERS",
                                                                       coordinate_format="SAME_AS_INPUT")[0]

    # Calculate the total geodesic area by summarizing the 'poly_area' attribute
    sum_area = calculate.calculate_statistics(sd, poly_shapefile, "poly_area")

    return sum_area

def get_country_fraction_data(sd, cell_size):
    
    # Get the directory of the current script
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')
    
    cell_size_str = str(cell_size)
    cntry_ds = None
    ds_grid = None

    if cell_size_str == "1" or cell_size_str == "1.0":
        cntry_ds = xr.load_dataset(config_dir + sd.config['file_paths']['Country_Fraction.1deg'])
        ds_grid = xr.load_dataset(config_dir + sd.config['file_paths']['grid_area_1deg'])
        ds_grid = ds_grid["grid_area_1deg"]
    elif cell_size_str == "0.5":
        cntry_ds = xr.load_dataset(config_dir + sd.config['file_paths']['Country_Fraction.0_5deg'])
        ds_grid = xr.load_dataset(config_dir + sd.config['file_paths']['grid_area_0_5deg'])
        ds_grid = ds_grid["grid_area_0_5deg"]

    return cntry_ds, ds_grid
