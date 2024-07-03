import os
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
import process_data
import calculate
import create
import sys
import json



def to_tif(sd, netcdf_path, netcdf_variable, out_tif_path, time):
    if time is not None:
        formatted_time = process_data.change_datetime_format(time)
        raster_layer = str(netcdf_variable) + "_" + time[:4]
        # Create a multidimensional raster layer with specified time
        sd.arcpy.md.MakeMultidimensionalRasterLayer(
            in_multidimensional_raster=netcdf_path,
            out_multidimensional_raster_layer=raster_layer,
            variables=[netcdf_variable], dimension_def="BY_VALUE",
            dimension_ranges=[],
            dimension_values=[["StdTime", formatted_time]],
            dimension="", dimensionless="NO_DIMENSIONS", spatial_reference="")
        output_raster = out_tif_path + raster_layer + ".tif"
        # Save the raster layer to a TIFF file
        sd.arcpy.management.CopyRaster(raster_layer, out_rasterdataset=output_raster)
    else:
        # Create a raster layer without a specific time
        raster_layer = str(netcdf_variable)
        sd.arcpy.md.MakeMultidimensionalRasterLayer(in_multidimensional_raster=netcdf_path,
                                                    out_multidimensional_raster_layer=raster_layer,
                                                    variables=[netcdf_variable])

        output_raster = out_tif_path + raster_layer + ".tif"
        sd.arcpy.management.CopyRaster(raster_layer, out_rasterdataset=output_raster)
    return output_raster


def from_tif(sd, in_tif_path, temp_path, short_name, long_name, units, source, cell_size, time, zero_is_value,
             conversion_type="SUM", verbose=False, value_per_sqm=None):
    # Open the input raster using sd.arcpy
    raster = sd.arcpy.Raster(in_tif_path)

    cell_size = float(cell_size)

    check_cell_size(sd, raster)
    # Extract the base filename from the input raster path
    base_filename = os.path.splitext(os.path.basename(in_tif_path))[0]

    # Set the extent for geographic coordinates
    with sd.arcpy.EnvManager(extent="-180 -90 180 90 GEOGCS[GCS_WGS_1984,DATUM[D_WGS_1984,SPHEROID[WGS_1984,6378137.0,298.257223563]],PRIMEM[Greenwich,0.0],UNIT[Degree,0.0174532925199433]]"):
        # Generate the output raster path
        output_raster = temp_path + base_filename + "_filled.tif"
        # Fill Nodata values with 0
        filled_raster = sd.arcpy.sa.Con(sd.arcpy.sa.IsNull(raster), 0, raster)
        filled_raster.save(output_raster)

    # Open the filled raster using rasterio
    filled_raster = temp_path + base_filename + "_filled.tif"
    with rasterio.open(filled_raster) as src:
        arr = src.read(1, masked=True).filled(0).astype(np.float64)

        raw_global_value = calculate.calculate_raw_global_value(conversion_type, arr, zero_is_value)

        # Calculate grid resolution
        num_lat, num_lon = calculate.calculate_grid_resolution(cell_size)
        padded_arr = calculate.calculate_padded_array(num_lat, num_lon, arr)
        aligned_arr = calculate.calculate_aligned_array(num_lat, num_lon, padded_arr)

        # Calculate the re-gridded global value based on the conversion type
        regridded_global_value, da = calculate.calculate_regridded_global_value(conversion_type, num_lat, num_lon,
                                                                                aligned_arr, cell_size, zero_is_value)
        if verbose:
            print(f"Raw global {conversion_type}: {raw_global_value:.3f}")
            print(f"Re-gridded global {conversion_type}: {regridded_global_value:.3f}")

        ds = calculate.calculate_ds(sd, value_per_sqm, da, short_name, long_name, units, source, cell_size, time,
                                    zero_is_value)
    return ds


def load_country_fraction_dataset(sd, cell_size):
    # Get the directory of the current script
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

    if cell_size == "1" or cell_size == "1.0":
        country_ds = xr.load_dataset(sd.config['file_paths']['Country_Fraction.1deg'])
    elif cell_size == "0.5":
        country_ds = xr.load_dataset(sd.config['file_paths']['Country_Fraction.0_5deg'])
    else:
        raise ValueError("The netcdf variable should be either 1, 0.5 or 0.25 degrees in resolution.")
    return country_ds


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


def normalize_by_area(sd, cell_size, da):
    # Get the directory of the current script
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')
    
    if cell_size == "1" or cell_size == "1.0":
        grid_ds = xr.load_dataset(sd.config['file_paths']['grid_area_1deg'])
        da = da / grid_ds["grid_area_1deg"]
    elif cell_size == "0.5":
        grid_ds = xr.load_dataset(sd.config['file_paths']['grid_area_0_5deg'])
        da = da / grid_ds["grid_area_0_5deg"]
    elif cell_size == "0.25":
        grid_ds = xr.load_dataset(sd.config['file_paths']['grid_area_0_25deg'])
        da = da / grid_ds["grid_area_0_25deg"]
    return da


def regrid(sd, raster_path, temp_path):
    # Open the raster dataset using sd.arcpy
    raster = sd.arcpy.Raster(raster_path)
    base_filename = os.path.splitext(os.path.basename(raster_path))[0]

    # Get the spatial reference of the raster
    ref = sd.arcpy.Describe(raster).SpatialReference
    # Get the name of the coordinate system
    coord_sys_name = sd.arcpy.env.outputCoordinateSystem.name

    # Check if the coordinate system is unknown and define the coordinate system as GCS WGS 1984 if True
    if ref.name == "Unknown":
        raster = sd.arcpy.management.DefineProjection(raster)
        print(f"The coordinate system of {base_filename} is defined to {coord_sys_name}.")
    # Check if coordinate system is already the target coordinate system; if True, print the spatial reference status
    elif ref.name == coord_sys_name:
        print(f"The coordinate sytem of {base_filename} is already defined to {coord_sys_name}.")
    # If a temporary path is not provided, reproject the raster to the target coordinate system
    elif ref.name != coord_sys_name:
        # temp_path = self._create_temp_folder(raster_path, "temp") + "/"
        print(f"Reprojecting the raster file.")

        filename = os.path.basename(raster_path)
        with sd.arcpy.EnvManager(
                extent="-180 -90 180 90 GEOGCS[GCS_WGS_1984,DATUM[D_WGS_1984,SPHEROID[WGS_1984,6378137.0,298.257223563]],PRIMEM[Greenwich,0.0],UNIT[Degree,0.0174532925199433]]"):
            raster = sd.arcpy.management.ProjectRaster(raster, temp_path + filename,
                                                    sd.arcpy.env.outputCoordinateSystem, "NEAREST", "", "", "#", "#")
            raster_path = temp_path + filename
            print(f"The raster file has been reprojected to {coord_sys_name}.")
    else:
        print(f"Projection type is not recognized!")
    return raster_path


def da_to_ds(sd, da, short_name, long_name, units, source, time, cell_size, zero_is_value):
    # Convert the DataArray to a Dataset with the specified variable name
    ds = da.to_dataset(name=short_name)

    # Add time dimension to dataset if provided and not set to 'recent'
    if time and time != 'recent':
        time_d = pd.to_datetime(time)
        ds = ds.assign_coords(time=time_d)
        ds = ds.expand_dims(dim='time')

    # Set variable attributes including short name, long name, units, and source if available
    attrs = {'short_name': short_name, 'long_name': long_name, 'units': units}
    if source is not None:
        attrs['source'] = source

    ds[short_name].attrs = attrs
    ds['lat'].attrs, ds['lon'].attrs = get_lat_lon_attrs()

    if not zero_is_value or zero_is_value.upper() != "YES":
        # Replace the 0 values to NaN, where zero shows evidence of absence.
        ds[short_name] = ds[short_name].where(ds[short_name] != 0, np.nan)

    ds = merge_with_grid_area(sd, ds, cell_size)
    return ds


def regridded_tif_2_ds(sd, raster_path, short_name, long_name, units, source, time, cell_size, zero_is_value=False):
    ds = xr.open_dataset(raster_path)  # Open into an xarray.DataArray
    ds = ds.astype(np.float64).isel(band=0).drop_vars(['band', 'spatial_ref']).rename({'x': 'lon', 'y': 'lat'})

    # Round latitude and longitude values to one decimal place
    decimal_places = 3  # Adjust as needed
    ds['lon'] = ds['lon'].round(decimal_places)
    ds['lat'] = ds['lat'].round(decimal_places)
    ds['lat'].attrs, ds['lon'].attrs = get_lat_lon_attrs()

    # change the variable name from default band_data to specified name
    ds = ds.rename({'band_data': short_name})

    # Add time dimension if provided
    if time is not None:
        time_d = pd.to_datetime(time)
        ds = ds.assign_coords(time=time_d)
        ds = ds.expand_dims(dim='time')

    # Add variable metadata
    attrs = {'short_name': short_name, 'long_name': long_name, 'units': units}
    if source is not None:
        attrs['source'] = source

    ds[short_name].attrs = attrs

    if not zero_is_value or zero_is_value.upper() != "YES":
        # Replace the 0 values to NaN, where zero shows evidence of absence.
        ds[short_name] = ds[short_name].where(ds[short_name] != 0, np.nan)
    
    ds = merge_with_grid_area(sd, ds, cell_size)
    return ds


def merge_ds_list(sd, dataset_list, netcdf_path=None, filename=None):
    ds = xr.merge(dataset_list)
    ds.attrs = {}  # Delete autogenerated global attributes
    ds.attrs.update(sd.global_attr)  # Adding new global attributes

    if netcdf_path is not None:
        ds.to_netcdf(netcdf_path + filename + ".nc")
    return ds


def move_points(sd, input_shapefile, world_shp, output_shapefile, x_offset=0.0001, y_offset=0.0001):
    point_selected, output_layer_names, num_records = sd.arcpy.management.SelectLayerByLocation(input_shapefile,
                                                                                                "BOUNDARY_TOUCHES",
                                                                                                world_shp, "",
                                                                                                "NEW_SELECTION",
                                                                                                "NOT_INVERT")
    # Create a copy of the input shapefile to store the modified points
    sd.arcpy.CopyFeatures_management(point_selected, output_shapefile)

    # Update the copied shapefile by moving points by the specified offsets
    with sd.arcpy.da.UpdateCursor(output_shapefile, ["SHAPE@XY"]) as cursor:
        for row in cursor:
            x, y = row[0]
            new_x = x + x_offset
            new_y = y + y_offset
            row[0] = (new_x, new_y)
            cursor.updateRow(row)

    return output_shapefile, point_selected


def point_2_tif(sd, input_shapefile, temp_path, gdb_path, cell_size, filename, field_name=None, field_summary=None):
    # Create a world shapefile for summarization
    cell_size_name = process_data.replace_special_characters(str(cell_size))
    world_shp = temp_path + "World_" + cell_size_name + "deg.shp"

    # Set a default summary statistic if not provided
    if field_summary is None:
        field_summary = 'SUM'

    # Define output file name based on provided or default values
    if field_name is not None:
        if filename is not None:
            outname = "Point_" + filename + field_name + "_" + field_summary
            output_file = temp_path + outname + ".tif"
        else:
            outname = "Point_" + field_name + "_" + field_summary
            output_file = temp_path + outname + ".tif"

        # Summarize points within polygons and create a raster
        summarize = sd.arcpy.analysis.SummarizeWithin(world_shp, input_shapefile, gdb_path + outname, "KEEP_ALL",
                                                        [[field_name, field_summary]], "ADD_SHAPE_SUM",
                                                        "SQUAREKILOMETERS", "", "NO_MIN_MAJ", "NO_PERCENT", "")

        value_field = field_summary + "_" + field_name
        sd.arcpy.conversion.PolygonToRaster(summarize, value_field, output_file, "CELL_CENTER", "NONE",
                                              cell_size, "DO_NOT_BUILD")
    else:
        # Define output file name for point counts
        if filename is not None:
            outname = filename + "Point_Counts"
        else:
            outname = "Point_Counts"

        output_file = temp_path + outname + ".tif"

        # Summarize points within polygons and create a raster for point counts
        summarize = sd.arcpy.analysis.SummarizeWithin(world_shp, input_shapefile, gdb_path + outname, "KEEP_ALL",
                                                        [], "ADD_SHAPE_SUM", "SQUAREKILOMETERS", "", "NO_MIN_MAJ",
                                                        "NO_PERCENT", "")

        sd.arcpy.conversion.PolygonToRaster(summarize, "Point_Count", output_file, "CELL_CENTER", "NONE",
                                              cell_size, "DO_NOT_BUILD")

    return output_file


def line_2_tif(sd, input_shapefile, temp_path, gdb_path, cell_size, filename, field_name=None, field_summary=None):
    world_shp = create.create_gridded_polygon(sd, temp_path, cell_size)
    cell_size_name = process_data.replace_special_characters(str(cell_size))
    world_shp = temp_path + "World_" + cell_size_name + "deg.shp"

    if field_summary is None:
        field_summary = 'SUM'

    if field_name is not None:
        # Generate output name based on field_name and filename
        if filename is not None:
            outname = "Line_" + filename + field_name + "_" + field_summary
            output_file = temp_path + outname + ".tif"
        else:
            outname = "Line_" + field_name + "_" + field_summary
            output_file = temp_path + outname + ".tif"

        # Use SummarizeWithin to calculate statistics within polygons
        summarize = sd.arcpy.analysis.SummarizeWithin(world_shp, input_shapefile, gdb_path + outname, "KEEP_ALL",
                                                        [[field_name, field_summary]], "ADD_SHAPE_SUM",
                                                        "KILOMETERS", "", "NO_MIN_MAJ", "NO_PERCENT", "")

        # Define the value field based on field_name and field_summary
        value_field = field_summary + "_" + field_name

        # Convert summarized data to raster
        sd.arcpy.conversion.PolygonToRaster(summarize, value_field, output_file, "CELL_CENTER", "NONE",
                                            cell_size, "DO_NOT_BUILD")
    else:
        # Generate output name based on filename
        if filename is not None:
            outname = filename + "Line_length"
            output_file = temp_path + outname + ".tif"
        else:
            outname = "Line_length"
            output_file = temp_path + outname + ".tif"

        # Use SummarizeWithin to calculate line lengths within polygons
        summarize = sd.arcpy.analysis.SummarizeWithin(world_shp, input_shapefile, gdb_path + outname, "KEEP_ALL",
                                                      [], "ADD_SHAPE_SUM", "KILOMETERS", "", "NO_MIN_MAJ",
                                                      "NO_PERCENT", "")

        # Convert summarized line lengths to raster
        sd.arcpy.conversion.PolygonToRaster(summarize, "sum_Length_KILOMETERS", output_file, "CELL_CENTER",
                                            "NONE", cell_size, "DO_NOT_BUILD")

    return output_file


def poly_2_tif(sd, input_shapefile, temp_path, gdb_path, cell_size, filename, fraction=None):
    cell_size_name = process_data.replace_special_characters(str(cell_size))
    world_shp = temp_path + "World_" + cell_size_name + "deg.shp"

    # Determine the output file name
    if filename is not None:
        outname = "Summarize_" + filename
        output_file = temp_path + outname + ".tif"
    else:
        outname = "Summarize_Polygon"
        output_file = temp_path + outname + ".tif"

    # Summarize area within grid cells
    summarize = sd.arcpy.analysis.SummarizeWithin(world_shp, input_shapefile, gdb_path + outname, "KEEP_ALL", [],
                                                    "ADD_SHAPE_SUM", "SQUAREKILOMETERS", "", "NO_MIN_MAJ",
                                                    "NO_PERCENT", "")

    # Check if fraction data needs to be calculated
    if fraction and fraction.upper() == "YES":
        # Calculate fraction field and create raster
        fraction = sd.arcpy.management.CalculateField(summarize, "frac", "!sum_Area_SQUAREKILOMETERS! / !g_area!",
                                                        "PYTHON3", "", "TEXT", "NO_ENFORCE_DOMAINS")[0]

        raster = sd.arcpy.conversion.PolygonToRaster(fraction, "frac", output_file, "CELL_CENTER", "NONE",
                                                       cell_size, "DO_NOT_BUILD")

        # Calculate and print the global sum of area
        value = calculate.calculate_statistics(sd, fraction, "sum_Area_SQUAREKILOMETERS")
        print(f"Global sum of area after gridding : {value:.2f} km2.")
        print("Fraction is created.")
    else:
        # Create raster without fraction data
        raster = sd.arcpy.conversion.PolygonToRaster(summarize, "sum_Area_SQUAREKILOMETERS", output_file,
                                                       "CELL_CENTER", "NONE", cell_size,
                                                       "DO_NOT_BUILD")

    return output_file


def check_cell_size(sd, raster):
    # Get the X and Y cell sizes from the raster properties, and round and convert cell sizes to float
    x_size = round(float(sd.arcpy.management.GetRasterProperties(raster, 'CELLSIZEX').getOutput(0)), 3)
    y_size = round(float(sd.arcpy.management.GetRasterProperties(raster, 'CELLSIZEY').getOutput(0)), 3)

    if x_size != y_size:
        raise ValueError("X and Y cell size are not equal!")

    print(f"X and Y-cell size of raw data: {x_size} degree")


def get_lat_lon_attrs():
    # add lat and lon attributes
    lat_attr = {"units": "degrees_north",
                "point_spacing": "even",
                "axis": "Y"}

    lon_attr = {"units": "degrees_east",
                "modulo": 360.,
                "point_spacing": "even",
                "axis": "X"}
    return lat_attr, lon_attr


def merge_with_grid_area(sd, ds, cell_size):
    
    # Get the directory of the current script
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

    # Merging the generated dataset with grid area
    cell_size_str = str(cell_size)
    if cell_size_str == "1" or cell_size_str == "1.0":
        ds = xr.merge([ds, xr.load_dataset(sd.config['file_paths']['grid_area_1deg'])])
    elif cell_size_str == "0.5":
        ds = xr.merge([ds, xr.load_dataset(sd.config['file_paths']['grid_area_0_5deg'])])
    elif cell_size_str == "0.25":
        ds = xr.merge([ds, xr.load_dataset(sd.config['file_paths']['grid_area_0_25deg'])])

    # Set and add global attributes
    ds.attrs = sd.global_attr
    return ds


def print_global_gridded_val(fold_function, ds_var):
    if fold_function.upper() == "SUM":
        print(f"Global gridded {fold_function}: {ds_var.sum().item()}")
    elif fold_function.upper() == "MEAN":
        print(f"Global gridded {fold_function}: {ds_var.mean().item()}")
    elif fold_function.upper() == "MAX":
        print(f"Global gridded {fold_function}: {ds_var.max().item()}")
    elif fold_function.upper() == "STD":
        print(f"Global gridded {fold_function}: {ds_var.std().item()}")
    else:
        sys.exit(1)
    return None


def adjust_time_var(time, ds, cntry_ds, var):
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

    return cntry_ds, ds_var


def add_year_col(time, dataframes):
    if time is not None:
        year = time[:4]
        merged_df = dataframes[0]  # Start with the first DataFrame
        merged_df = pd.concat([merged_df['ISO3'], pd.DataFrame({'Year': year}, index=merged_df.index),
                               merged_df.drop(columns=['ISO3'])], axis=1)
    else:
        merged_df = dataframes[0]  # Start with the first DataFrame
    return merged_df


def save_to_nc(output_directory, output_filename, time, ds, base_filename):
    if output_directory != None:
        if time != None:
            if output_filename != None:
                ds.to_netcdf(output_directory + output_filename + "_" + str(time) + ".nc")
            else:
                ds.to_netcdf(output_directory + base_filename + "_" + str(time) + ".nc")
        else:
            if output_filename != None:
                ds.to_netcdf(output_directory + output_filename + ".nc")
            else:
                ds.to_netcdf(output_directory + base_filename + ".nc")
                
                
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
