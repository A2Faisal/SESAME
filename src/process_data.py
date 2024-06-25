import os
import re
import xarray as xr
from datetime import datetime
import utils
import create


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


def identify_lat_lon_names(netcdf_path):
    ds = xr.open_dataset(netcdf_path)

    common_lat_names = [
        'lat',
        'latitude',
        'y',
        'south_north',
        'grid_latitude',
        'latitudes',
        'Y',
        'Y_AXIS'
    ]

    common_lon_names = [
        'lon',
        'longitude',
        'x',
        'west_east',
        'grid_longitude',
        'longitudes',
        'X',
        'X_AXIS'
    ]

    # Identify latitude and longitude coordinates
    lat_coord = next((coord for coord in ds.coords if coord in common_lat_names), None)
    lon_coord = next((coord for coord in ds.coords if coord in common_lon_names), None)

    # Determine x and y dimensions based on latitude and longitude coordinates
    x_dimension = lon_coord
    y_dimension = lat_coord

    return x_dimension, y_dimension


def check_shapefile_projection(sd, shapefile_path, temp_path=None):
    shapefile = shapefile_path
    # get the base file name
    base_filename = os.path.splitext(os.path.basename(shapefile))[0]

    # Get the spatial reference of the raster
    ref = sd.arcpy.Describe(shapefile_path).SpatialReference
    # Get the name of the coordinate system
    coord_sys_name = sd.arcpy.env.outputCoordinateSystem.name

    # Check if the coordinate system is unknown and define the coordinate system as GCS WGS 1984 if True
    if ref.name == "Unknown":
        shapefile = sd.arcpy.management.DefineProjection(shapefile)
        print(f"The coordinate system of {base_filename} is defined to {coord_sys_name}.")
    # Check if coordinate system is already the target coordinate system; if True, print the spatial reference status
    elif ref.name == coord_sys_name:
        print(f"The coordinate sytem of {base_filename} is already defined to {coord_sys_name}.")
    # If a temporary path is not provided, reproject the raster to the target coordinate system
    elif ref.name != coord_sys_name:
        temp_path = create.create_temp_folder(input_path=shapefile_path, folder_name="temp")
        temp_path = temp_path + "/"
        print(f"Reprojecting the shapefile.")
        filename = os.path.basename(shapefile_path)
        with sd.arcpy.EnvManager(
                extent="-180 -90 180 90 GEOGCS[GCS_WGS_1984,DATUM[D_WGS_1984,SPHEROID[WGS_1984,6378137.0,298.257223563]],PRIMEM[Greenwich,0.0],UNIT[Degree,0.0174532925199433]]"):
            shapefile = sd.arcpy.management.Project(shapefile, temp_path + filename,
                                                      sd.arcpy.env.outputCoordinateSystem)
            shapefile_path = temp_path + filename
            print(f"The shapefile has been reprojected to {coord_sys_name}.")
    else:
        print(f"Projection type is not recognized!")

    return shapefile_path


def change_datetime_format(original_date):
    # Split the time string into seconds and microseconds
    date, microseconds = original_date.split('.')

    # Convert seconds part to datetime object
    original_datetime = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')

    # Add microseconds separately
    original_datetime = original_datetime.replace(microsecond=int(microseconds))

    # Format the datetime object
    formatted_time_value = original_datetime.strftime('%Y-%m-%dT%H:%M:%S')

    return formatted_time_value


def adjust_shapefile_points(sd, input_shapefile, world_shp, output_path):
    base_filename = os.path.splitext(os.path.basename(input_shapefile))[0]
    copy_shapefile = output_path + base_filename + "_copy.shp"
    sd.arcpy.management.CopyFeatures(input_shapefile, copy_shapefile, "", None, None, None)
    output_shapefile = output_path + base_filename + "_temp.shp"

    point_adjusted, point_selected = utils.move_points(sd, copy_shapefile, world_shp, output_shapefile)
    # Process: Delete Rows
    input_with_deleted_rows = sd.arcpy.management.DeleteRows(in_rows=point_selected)[0]

    final_shapefile = output_path + base_filename + "_adj.shp"
    # Merge the modified selected points with the original input shapefile
    sd.arcpy.Merge_management([copy_shapefile, point_adjusted], final_shapefile)

    return final_shapefile
