import os
import xarray as xr
import process_data
import get
import utils
import calculate
import pandas as pd


def create_new_ds(sd, input_ds, tabular_column, country_ds, netcdf_variable, input_df, verbose):
    country_netcdf = country_ds * input_ds[netcdf_variable]
    new_ds = xr.Dataset(coords=input_ds.coords)

    for var_name in country_netcdf.variables:
        if var_name in input_df["ISO3"].values:
            # Get the corresponding Numeric value from the DataFrame
            numeric_value = input_df.loc[input_df["ISO3"] == var_name, tabular_column].values[0]
            total_country = country_netcdf[var_name].sum().item()
            if numeric_value > 0 and total_country == 0:
                country_ds_copy = country_ds[var_name].copy()
                netcdf_da = xr.where(country_ds_copy != 0, 1, country_ds_copy)
                new_country_netcdf = country_ds_copy * netcdf_da
                new_country_netcdf = new_country_netcdf.to_dataset()
                total_country = new_country_netcdf[var_name].sum().item()
                new_ds[var_name] = (new_country_netcdf[var_name] * numeric_value) / total_country
                if verbose:
                    print(f"{var_name} evenly distributed.")
            else:
                # Dasymmetric equation
                new_ds[var_name] = (country_netcdf[var_name] * numeric_value) / total_country
    return new_ds


def create_temp_folder(input_path, folder_name="temp"):
    parent_dir = os.path.dirname(os.path.dirname(input_path))
    path = os.path.join(parent_dir, folder_name)

    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, '')
    return path


def create_gridded_polygon(sd, raster_path, cell_size, polygon_path=None, grid_area=None, verbose=False):
    if polygon_path is None:
        # creating temporary folder
        temp_path = create_temp_folder(input_path=raster_path, folder_name="temp")
        polygon_path = temp_path

    # Create world gridded layer
    cell_size_name = process_data.replace_special_characters(str(cell_size))
    filename = "World_" + cell_size_name + "deg.shp"
    world_shp = sd.arcpy.management.CreateFishnet(polygon_path + filename, "-180 -90", "-180 -80",
                                                  cell_size, cell_size, None, None,
                                                  "180 90", "NO_LABELS",
                                                  "-180 -90 180 90 GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]]",
                                                  "POLYGON")[0]

    # Adding attributes for ID, grid area, and fraction
    world_shp = \
        sd.arcpy.management.AddField(world_shp, "id", "LONG", None, None, None, "", "NULLABLE", "NON_REQUIRED", "")[0]

    world_shp = sd.arcpy.management.AddField(world_shp, "g_area", "DOUBLE", None, None, None, "", "NULLABLE",
                                             "NON_REQUIRED", "")[0]

    world_shp = sd.arcpy.management.AddField(world_shp, "frac", "DOUBLE", None, None, None, "", "NULLABLE",
                                             "NON_REQUIRED", "")[0]

    # Generate ID numbers based on FID
    world_shp = sd.arcpy.management.CalculateField(world_shp, "id", "!FID!+1", "PYTHON3", "", "TEXT",
                                                   "NO_ENFORCE_DOMAINS")[0]

    if grid_area is not None:
        # Calculating the area of each grid
        world_shp = polygon_path + filename
        world_shp = sd.arcpy.management.CalculateGeometryAttributes(world_shp, [["g_area", "AREA_GEODESIC"]],
                                                                    area_unit="SQUARE_KILOMETERS",
                                                                    coordinate_system="GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]]")[
            0]

    if verbose:
        print("Global gridded polygon is created.")

    return world_shp


def create_df(sd, dataframes, var, verbose, cell_size, grid_area, fold_function, time, ds):
    # Select the country fraction data based on cell size
    try:
        cntry_ds, ds_grid = get.get_country_fraction_data(sd, cell_size)
    except FileNotFoundError as e:
        print(f" Error while reading file {e} ")

    # Check if 'time' is a dimension in the variable
    cntry_ds, ds_var = utils.adjust_time_var(time, ds, cntry_ds, var)

    if grid_area is not None and grid_area.upper() == 'YES':
        # Multiply the variable by grid area if specified
        ds_var = ds_var * ds_grid
        print(f"Generating the tabular data for: {var}")

    if verbose:
        utils.print_global_gridded_val(fold_function, ds_var)

    ds_var = ds_var * cntry_ds

    data = calculate.calculate_fold_function(fold_function, ds_var)

    # Extract variable names and values
    variable_names = list(data.data_vars.keys())
    values = [data[var].values.item() for var in variable_names]

    # Create a DataFrame from variable names and values
    df = pd.DataFrame({'ISO3': variable_names, var: values})
    dataframes.append(df)

    return dataframes
