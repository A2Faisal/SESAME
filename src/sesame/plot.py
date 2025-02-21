import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import cartopy
import seaborn as sns

import calculate


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
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.open_dataset(netcdf_directory)    
    
    # Ensure the specified variable is in the dataset
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
        
    data = dataset[variable].values.flatten()

    # Remove NaNs
    data = data[~np.isnan(data)]

    # Remove outliers if specified
    if remove_outliers:
            # Calculate the mean and standard deviation
            mean = np.mean(data)
            std_dev = np.std(data)
            # Calculate Z-scores
            z_scores = (data - mean) / std_dev
            # Define a threshold for Z-score (e.g., 3)
            threshold = 3
            # Filter data within the threshold
            data = data[(np.abs(z_scores) <= threshold)]

    # Apply log transformation if specified
    if log_transform:
        if log_transform == 'log10':
            data = np.log10(data)
        elif log_transform == 'log':
            data = np.log(data)
        elif log_transform == 'log2':
            data = np.log2(data)
        else:
            raise ValueError(f"Unsupported log transform '{log_transform}'. Use 'log10', 'log', or 'log2'.")
    
    # Create the histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=bin_size, kde=True, color=color)
    plt.title(plot_title, fontsize=16)
    if x_label:
        plt.xlabel(x_label, fontsize=14)
    else:
        plt.xlabel(variable)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(False)
    
    if output_dir:
        if filename:
            plt.savefig(output_dir + filename, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(output_dir + "output_histogram.png", dpi=600, bbox_inches='tight')
            
    plt.show()
    

    
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
    
    # Check and load dataset for variable1
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'dataset' or 'netcdf_directory' must be provided for variable1.")
    elif dataset is None:
        dataset = xr.open_dataset(netcdf_directory)
    elif netcdf_directory is not None:
        raise ValueError("Only one of 'dataset' or 'netcdf_directory' should be provided for variable1.")
    
    # Check and load dataset2 for variable2, or allow variable2 to come from dataset
    if dataset2 is None and netcdf_directory2 is None:
        dataset2 = dataset  # If no second dataset or directory is provided, use the same dataset
    elif dataset2 is None:
        dataset2 = xr.open_dataset(netcdf_directory2)
    elif netcdf_directory2 is not None:
        raise ValueError("Only one of 'dataset2' or 'netcdf_directory2' should be provided for variable2.")
    
    # Ensure both variables exist in their respective datasets
    if variable1 not in dataset:
        raise ValueError(f"Variable '{variable1}' not found in the dataset.")
    
    if variable2 not in dataset2:
        raise ValueError(f"Variable '{variable2}' not found in the second dataset.")
    
    # Get data for the x-axis
    data1 = dataset[variable1].values.flatten()

    # Get data for the y-axis from either dataset or dataset2
    if dataset2 is None:
        # If dataset2 is not provided, use the same dataset for both axes
        if variable2 not in dataset:
            raise ValueError(f"Variable '{variable2}' not found in the dataset.")
        data2 = dataset[variable2].values.flatten()
    else:
        # If dataset2 is provided, use dataset2 for the y-axis
        if variable2 not in dataset2:
            raise ValueError(f"Variable '{variable2}' not found in dataset2.")
        data2 = dataset2[variable2].values.flatten()
    
    # Create a DataFrame from the data
    df = pd.DataFrame({
        variable1 : data1,
        variable2 : data2
    })
    
    # Combine data into a DataFrame
    df = pd.DataFrame({variable1: data1, variable2: data2})
    # Replace 0 values with NaN
    df.replace(0, np.nan, inplace=True)
    df = df.dropna()
    

    # Apply log transformation if specified
    if log_transform_1:
        if log_transform_1 == 'log10':
            df[variable1] = np.log10(df[variable1])
        elif log_transform_1 == 'log':
            df[variable1] = np.log(df[variable1])
        elif log_transform_1 == 'log2':
            df[variable1] = np.log2(df[variable1])
        else:
            raise ValueError(f"Unsupported log transform '{log_transform_1}'. Use 'log10', 'log', or 'log2'.")

    if log_transform_2:
        if log_transform_2 == 'log10':
            df[variable2] = np.log10(df[variable2])
        elif log_transform_2 == 'log':
            df[variable2] = np.log(df[variable2])
        elif log_transform_2 == 'log2':
            df[variable2] = np.log2(df[variable2])
        else:
            raise ValueError(f"Unsupported log transform '{log_transform_2}'. Use 'log10', 'log', or 'log2'.")

    # Remove outliers if specified
    if remove_outliers:
        # Calculate the mean and standard deviation
        mean = df.mean()
        std_dev = df.std()
        # Calculate Z-scores
        z_scores = (df - mean) / std_dev
        # Define a threshold for Z-score (e.g., 3)
        threshold = 3
        # Create a boolean mask for data within the threshold
        without_outliers = (z_scores.abs() <= threshold).all(axis=1)
        # Filter the DataFrame to remove outliers
        df = df[without_outliers]
        
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=variable1, y=variable2, data=df, color=color)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x=df[variable1], y=df[variable2])
    
    # Draw the regression line
    sns.lineplot(x=df[variable1], y=slope * df[variable1] + intercept, color='blue', ax=scatter)

    if equation:
        # Add the equation to the plot
        plt.text(0.1, 0.95, f'y = {slope:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes)
        # Add the p-value to the plot
        plt.text(0.1, 0.9, f'P-value: {p_value:.2f}', transform=plt.gca().transAxes)    


    # Add labels to the x and y axes
    if x_label:
        plt.xlabel(x_label, fontsize=14)
    else:
        plt.xlabel(variable1, fontsize=14)
        
    if y_label:
        plt.ylabel(y_label, fontsize=14)
    else:
        plt.ylabel(variable2, fontsize=14)
    
    plt.title(plot_title, fontsize=16)
    # Show the plot
    plt.grid(False)
    
    if output_dir:
        if filename:
            plt.savefig(output_dir + filename, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(output_dir + "output_scatter.png", dpi=600, bbox_inches='tight')

    plt.show()
    
    

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
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.open_dataset(netcdf_directory)
        
    if dataset2 is None and netcdf_directory2 is None:
        raise ValueError("Either 'xarray dataset2' or 'netcdf_directory2' must be provided.")
    elif dataset2 is not None and netcdf_directory2 is not None:
        raise ValueError("Only one of 'xarray dataset2' or 'netcdf_directory2' should be provided.")
    
    if netcdf_directory:
        dataset = xr.open_dataset(netcdf_directory)
    
    # Ensure the specified variable for the x-axis is in the dataset
    if variable1 not in dataset:
        raise ValueError(f"Variable '{variable1}' not found in the dataset.")
    
    # Get data for the x-axis
    data1 = dataset[variable1].values.flatten()

    # Get data for the y-axis from either dataset or dataset2
    if dataset2 is None:
        # If dataset2 is not provided, use the same dataset for both axes
        if variable2 not in dataset:
            raise ValueError(f"Variable '{variable2}' not found in the dataset.")
        data2 = dataset[variable2].values.flatten()
    else:
        # If dataset2 is provided, use dataset2 for the y-axis
        if variable2 not in dataset2:
            raise ValueError(f"Variable '{variable2}' not found in dataset2.")
        data2 = dataset2[variable2].values.flatten()
    
    # Create a DataFrame from the data
    df = pd.DataFrame({
        variable1 : data1,
        variable2 : data2
    })
    
    # Combine data into a DataFrame
    df = pd.DataFrame({variable1: data1, variable2: data2})
    # Replace 0 values with NaN
    df.replace(0, np.nan, inplace=True)
    df = df.dropna()
    

    # Apply log transformation if specified
    if log_transform_1:
        if log_transform_1 == 'log10':
            df[variable1] = np.log10(df[variable1])
        elif log_transform_1 == 'log':
            df[variable1] = np.log(df[variable1])
        elif log_transform_1 == 'log2':
            df[variable1] = np.log2(df[variable1])
        else:
            raise ValueError(f"Unsupported log transform '{log_transform_1}'. Use 'log10', 'log', or 'log2'.")

    if log_transform_2:
        if log_transform_2 == 'log10':
            df[variable2] = np.log10(df[variable2])
        elif log_transform_2 == 'log':
            df[variable2] = np.log(df[variable2])
        elif log_transform_2 == 'log2':
            df[variable2] = np.log2(df[variable2])
        else:
            raise ValueError(f"Unsupported log transform '{log_transform_2}'. Use 'log10', 'log', or 'log2'.")

    # Remove outliers if specified
    if remove_outliers:
        # Calculate the mean and standard deviation
        mean = df.mean()
        std_dev = df.std()
        # Calculate Z-scores
        z_scores = (df - mean) / std_dev
        # Define a threshold for Z-score (e.g., 3)
        threshold = 3
        # Create a boolean mask for data within the threshold
        without_outliers = (z_scores.abs() <= threshold).all(axis=1)
        # Filter the DataFrame to remove outliers
        df = df[without_outliers]
        
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    
    # Create a hexbin plot
    plt.hexbin(df[variable1], df[variable2], gridsize=grid_size, cmap=color)

    # Add a colorbar
    plt.colorbar(label='count')


    # Add labels to the x and y axes
    if x_label:
        plt.xlabel(x_label, fontsize=14)
    else:
        plt.xlabel(variable1, fontsize=14)
        
    if y_label:
        plt.ylabel(y_label, fontsize=14)
    else:
        plt.ylabel(variable2, fontsize=14)
    
    plt.title(plot_title, fontsize=16)
    # Show the plot
    plt.grid(False)
    
    if output_dir:
        if filename:
            plt.savefig(output_dir + filename, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(output_dir + "output_hexbin.png", dpi=600, bbox_inches='tight')
    plt.show()



def plot_time_series(variable, dataset=None, agg_function='sum', plot_type='both', color='blue', plot_label='Area Plot', x_label='Year', y_label='Value', plot_title='Time Series Plot', smoothing_window=None, output_dir=None, filename=None, netcdf_directory=None):
    """
    Create a line plot and/or area plot for a time series data variable.
    
    Parameters:
    - ds: xarray.Dataset, the dataset containing the variable to plot.
    - variable: str, the name of the variable to plot.
    - agg_function: str, the operation to apply ('sum', 'mean', 'max', 'std').
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
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.open_dataset(netcdf_directory)  
    
    # Ensure the specified variable is in the dataset
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
    
    ds = dataset
    # Select the data variable
    data_var = ds[variable]

    # Perform the specified operation along the spatial dimensions
    if agg_function.lower() == 'sum':
        time_series = data_var.sum(dim=('lat', 'lon'))
    elif agg_function.lower() == 'mean':
        time_series = data_var.mean(dim=('lat', 'lon'))
    elif agg_function.lower() == 'max':
        time_series = data_var.max(dim=('lat', 'lon'))
    elif agg_function.lower() == 'std':
        time_series = data_var.std(dim=('lat', 'lon'))
    else:
        raise ValueError(f"Unsupported operation '{agg_function}'. Use 'sum', 'mean', 'max', or 'std'.")
    
    # Apply rolling mean smoothing if specified
    if smoothing_window:
        time_series = time_series.rolling(time=smoothing_window, min_periods=1).mean()

    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 6))

    if plot_type.lower() == 'line':
        ax.plot(time_series['time'], time_series.values, color=color, label=plot_label)
    
    if plot_type.lower() == 'area':
        ax.fill_between(time_series['time'], time_series.values, color=color, alpha=0.3, label=plot_label)
    
    if plot_type == 'both':
        ax.plot(time_series['time'], time_series.values, color=color)
        ax.fill_between(time_series['time'], time_series.values, color=color, alpha=0.3, label=plot_label)
    
    # Add labels to the x and y axes
    if x_label:
        plt.xlabel(x_label, fontsize=14)
    else:
        plt.xlabel(variable, fontsize=14)
        
    if y_label:
        plt.ylabel(y_label, fontsize=14)
    else:
        plt.ylabel(variable, fontsize=14)
    
    plt.title(plot_title, fontsize=16)
    
    ax.legend()
    
    if output_dir and filename:
        plt.savefig(output_dir + filename, dpi=600, bbox_inches='tight')
    elif filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    
    plt.show()


def plot_map(variable, dataset=None, color='hot_r', title='', label='', color_min=None, color_max=None, levels=10, output_dir=None, filename=None, netcdf_directory=None):
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.load_dataset(netcdf_directory) 

    if isinstance(levels, list):
        # Directly use provided levels if it's a list
        bounds = levels
        num_levels = len(levels) - 1
        # Ensure correct number of colors in the colormap
        cmap_discrete = plt.cm.get_cmap(color, num_levels)
        # Create a BoundaryNorm with the given bounds
        norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N, clip=False)
        
    else:
        # Use linear spacing if levels is an integer
        if color_min is None:
            color_min = dataset[variable].min().item()
        if color_max is None:
            color_max = dataset[variable].max().item()
        bounds = np.linspace(color_min, color_max, levels)
        bounds = np.round(bounds, 2)
        num_levels = levels
        # Ensure correct number of colors in the colormap
        cmap_discrete = plt.cm.get_cmap(color, num_levels)
        norm = Normalize(vmin=color_min, vmax=color_max)

    # Create a subplot with adjusted layout and aspect ratio
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(12, 6))

    # Plot the dataset
    im = ax.pcolormesh(
        dataset['lon'],
        dataset['lat'],
        dataset[variable].values,
        transform=ccrs.PlateCarree(),
        cmap=cmap_discrete,
        norm=norm,
    )

    ax.coastlines(resolution='110m', color='gray', linewidth=1)
    ax.add_feature(cfeature.LAND, color='white')
    ax.set_title(title)

    # Create a custom colorbar with a triangular arrow at the end
    cax = fig.add_axes([0.27, 0.03, 0.5, 0.05])  # Adjust these values to position the colorbar
    cb = ColorbarBase(cax, cmap=cmap_discrete, norm=norm, orientation='horizontal', extend='both' if np.min(bounds) < 0 else 'max')
    cb.set_label(label)
    
    if output_dir and filename:
        plt.savefig(output_dir + filename, dpi=600, bbox_inches='tight')
    plt.show()


def plot_country(column, dataframe=None, title="Map", label=None, color='viridis', num_classes=5, class_type='geometric', output_dir=None, filename=None, csv_path=None):
    if dataframe is None and csv_path is None:
        raise ValueError("Either 'pandas dataframe' or 'csv path' must be provided.")
    elif dataframe is not None and csv_path is not None:
        raise ValueError("Only one of 'pandas dataframe' or 'csv path' should be provided.")
    
    if csv_path:
        try:
            dataframe = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_path, encoding='latin1')
    
    # Load and project the world shapefile
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_directory, "data")
    shapefile_path =  os.path.join(data_dir, "CShapes_v2_converted_2023.shp")
    world_gdf = gpd.read_file(shapefile_path)
    world_gdf = world_gdf.to_crs('EPSG:4326')
    robinson_proj = ccrs.Robinson()
    world_gdf = world_gdf.to_crs(robinson_proj.proj4_init)

    # Merge the shapefile with the DataFrame
    merged = world_gdf.merge(dataframe, on='ISO3')

    # Classifier configuration
    if isinstance(class_type, str):
        if class_type == 'equal':
            merged['class'] = calculate.equal_interval(merged[column], num_classes)
        elif class_type == 'quantile':
            merged['class'] = calculate.quantile_classes(merged[column], num_classes)
        elif class_type == 'geometric':
            merged['class'] = calculate.geometric_interval(merged[column], num_classes)
        elif class_type == 'std':
            merged['class'] = calculate.standard_deviation(merged[column], num_classes)
    elif isinstance(class_type, list):  # Check if class_type is a list
        manual_bins = class_type
        merged['class'] = pd.cut(merged[column], bins=manual_bins, include_lowest=True, labels=False)
    else:
        raise ValueError("Invalid class_type provided. Must be 'equal', 'geometric', 'std_dev', or a list of bin edges.")

    # Calculate the actual number of unique classes present
    actual_classes = merged['class'].nunique()
    num_classes = min(num_classes, actual_classes)

    # Calculate the maximum value for the specified column within each class
    class_maxima = merged.groupby('class')[column].max()
    
    # Set up the plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(12, 6))
    ax.set_global()
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.75)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
    
    # Set colormap
    cmap = ListedColormap(sns.color_palette(color, num_classes).as_hex())

    # Plotting
    merged.plot(column='class', cmap=cmap, linewidth=0, ax=ax, edgecolor='0.8', legend=False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])  # Important for ensuring the colorbar recognizes the full range
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=12)
    
    # Assuming unique_classes is a list of numerical values
    colorbar_labels = [f"{int(x)}" for x in class_maxima]

    # Setting custom labels without tick marks
    tick_locations = np.linspace(0.5 / num_classes, 1 - 0.5 / num_classes, num_classes)
    # tick_locations = np.linspace(1 / num_classes, 1, num_classes)

    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels(colorbar_labels, fontsize=10)
    cbar.ax.set_xticklabels(colorbar_labels, rotation=0)
    cbar.ax.tick_params(size=0)
    
    # Add title and save the figure
    plt.title(title, fontsize=14)
    if output_dir and filename:
        plt.savefig(output_dir + filename, dpi=600, bbox_inches='tight')
    elif filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        
    plt.show()
