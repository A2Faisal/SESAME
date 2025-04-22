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


def plot_map(variable, dataset=None, color='hot_r', title='', label='',
             vmin=None, vmax=None, extend_min=False, extend_max=False, levels=10, out_bound=True, remove_ata=False,
             output_dir=None, filename=None, netcdf_directory=None, show=True):
    
    """
    Plots a 2D map of a variable from an xarray Dataset or NetCDF file with customizable colorbar, projection, and map appearance.

    Parameters
    ----------
    variable : str
        Name of the variable in the xarray Dataset to plot.
    dataset : xarray.Dataset, optional
        An already-loaded xarray Dataset containing the variable. Required if `netcdf_directory` is not provided.
    color : str, default 'hot_r'
        Matplotlib colormap name for the plot (discrete color scale).
    title : str, default ''
        Title of the map.
    label : str, default ''
        Label for the colorbar.
    vmin : float, optional
        Minimum data value for the colorbar range. If not provided, the minimum of the variable is used.
    vmax : float, optional
        Maximum data value for the colorbar range. If not provided, the maximum of the variable is used.
    extend_min : bool, default False
        If True, includes values below `vmin` in the first color class and shows a left arrow on the colorbar.
    extend_max : bool, default False
        If True, includes values above `vmax` in the last color class and shows a right arrow on the colorbar.
    levels : int or list of float, default 10
        Either the number of color intervals or a list of explicit interval boundaries.
    out_bound : bool, default True
        Whether to display the outer boundary (spine) of the map projection.
    remove_ata : bool, default False
        If True, removes Antarctica from the map by excluding data below 60°S latitude.
    output_dir : str, optional
        Directory path to save the output figure. If not provided, the figure is saved in the current working directory.
    filename : str, optional
        Filename (with extension) for saving the figure. If not provided, the plot is not saved.
    netcdf_directory : str, optional
        File path to a NetCDF file. Used if `dataset` is not provided.
    show: bool, True
        Whether or not show the map

    Notes
    -----
    - If both `extend_min` and `extend_max` are False, the dataset is clipped strictly within [vmin, vmax].
    - The colorbar will use arrows to indicate out-of-bound values only if `extend_min` or `extend_max` is True.
    - Tick formatting on the colorbar is:
        - Integer-only if the data range is all positive (vmin >= 0).
        - Two decimal places if any value is below 0.
    - If `remove_ata` is True, the colorbar is placed slightly higher to avoid overlap with the map.

    Raises
    ------
    ValueError
        If both or neither of `dataset` and `netcdf_directory` are provided.

    Example
    -------
    >>> plot_map(
    ...     variable='npp',
    ...     dataset=ds.isel(time=-1),
    ...     vmin=0,
    ...     vmax=1200,
    ...     extend_max=True,
    ...     color='Greens',
    ...     levels=10,
    ...     remove_ata=True,
    ...     title='Net Primary Productivity',
    ...     label='gC/m²/year',
    ...     filename='npp_map.png'
    ... )
    """
    
    if dataset is None and netcdf_directory is None:
        raise ValueError("Either 'xarray dataset' or 'netcdf_directory' must be provided.")
    elif dataset is not None and netcdf_directory is not None:
        raise ValueError("Only one of 'xarray dataset' or 'netcdf_directory' should be provided.")
    
    if netcdf_directory:
        dataset = xr.open_dataset(netcdf_directory) 

    data = dataset[variable]
    # Remove Antarctica if requested (e.g., keep only latitudes > -60°)
    if remove_ata:
        dataset = dataset.where(dataset['lat'] > -60, drop=True)
        data = dataset[variable]


    # Default vmin/vmax if not provided
    if vmin is None:
        vmin = data.min().item()
    if vmax is None:
        vmax = data.max().item()

    # Data filtering based on rounding flags
    extend = 'neither'
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
        data = data.where(data <= vmax)
    elif extend_max:
        extend = 'max'
        data = data.where(data >= vmin)
    else:
        data = data.where((data >= vmin) & (data <= vmax))

    # Create levels and colormap
    if isinstance(levels, list):
        bounds = levels
        num_levels = len(bounds) - 1
    else:
        step = (vmax - vmin) / levels
        bounds = np.arange(vmin, vmax + step, step)
        bounds = np.round(bounds, 2)
        num_levels = len(bounds) - 1

    # Updated colormap call (future-proof)
    cmap_discrete = plt.get_cmap(color, num_levels)
    
    # Color normalization
    norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

    # Plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(12, 6))
    im = ax.pcolormesh(
        dataset['lon'],
        dataset['lat'],
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap_discrete,
        norm=norm,
    )

    ax.coastlines(resolution='110m', color='gray', linewidth=1)
    ax.add_feature(cfeature.LAND, color='white')
    ax.set_title(title)
    ax.spines['geo'].set_visible(out_bound)

    # Colorbar
    if remove_ata:
        cax = fig.add_axes([0.27, 0.08, 0.5, 0.05])
    else:
        cax = fig.add_axes([0.27, 0.03, 0.5, 0.05])
        
    cb = ColorbarBase(cax, cmap=cmap_discrete, norm=norm, orientation='horizontal', extend=extend)
    cb.set_label(label)

    # Format colorbar ticks based on data range
    tick_values = bounds
    if (vmax - vmin) <= 10:
        tick_labels = [f"{val:.2f}" for val in tick_values]
    else:
        tick_labels = [f"{val:.0f}" for val in tick_values]

    cb.set_ticks(tick_values)
    cb.set_ticklabels(tick_labels)

    # Save figure
    if filename:
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    if show:
        plt.show()
        
    return ax
    


def plot_country(column, dataframe=None, title="", label="", color='viridis', levels=10, output_dir=None, filename=None, csv_path=None, 
                 remove_ata=False, out_bound=True, vmin=None, vmax=None, extend_min=False, extend_max=False):

    if dataframe is None and csv_path is None:
        raise ValueError("Provide either a dataframe or a csv_path.")
    if dataframe is not None and csv_path is not None:
        raise ValueError("Provide only one of dataframe or csv_path.")

    if csv_path:
        try:
            dataframe = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_path, encoding='latin1')

    if remove_ata:
        dataframe = dataframe[dataframe['ISO3'] != 'ATA']

    # Load shapefile
    # Load and project the world shapefile
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_directory, "data")
    shapefile_path =  os.path.join(data_dir, "CShapes_v2_converted_2023.shp")
    world_gdf = gpd.read_file(shapefile_path)
    world_gdf = world_gdf.to_crs('EPSG:4326')
    robinson_proj = ccrs.Robinson()
    world_gdf = world_gdf.to_crs(robinson_proj.proj4_init)

    # Merge with data
    merged = world_gdf.merge(dataframe, on='ISO3')
    data = merged[column]

    # --- Default vmin/vmax ---
    if vmin is None:
        vmin = data.min().item()
    if vmax is None:
        vmax = data.max().item()

    # --- Data masking based on flags ---
    extend = 'neither'
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
        data = data.where(data <= vmax)
    elif extend_max:
        extend = 'max'
        data = data.where(data >= vmin)
    else:
        data = data.where((data >= vmin) & (data <= vmax))

    # --- Create bounds using linspace (accurate binning) ---
    if isinstance(levels, list):
        bounds = levels
        num_levels = len(bounds) - 1
    else:
        bounds = np.linspace(vmin, vmax, levels + 1)
        bounds = np.round(bounds, 4)
        num_levels = len(bounds) - 1

    # --- Colormap & normalization ---
    cmap = plt.get_cmap(color, num_levels)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # --- Setup plot ---
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(12, 6))
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.spines['geo'].set_visible(out_bound)
    ax.set_title(title, fontsize=14)

    merged[column] = data
    merged.plot(column=column, cmap=cmap, norm=norm, linewidth=0, ax=ax, edgecolor='0.8',
                missing_kwds={"color": "lightgrey", "hatch": "///"})

    # --- Colorbar ---
    cax = fig.add_axes([0.27, 0.08 if remove_ata else 0.03, 0.5, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend=extend)
    cb.set_label(label, fontsize=12)

    # --- Ticks centered within each class ---
    tick_values = bounds
    show_decimals = abs(vmax - vmin) < 10

    if show_decimals:
        tick_labels = [f"{val:.2f}" for val in tick_values]
    else:
        tick_labels = [f"{val:.0f}" for val in tick_values]
        
    cb.set_ticks(tick_values)
    cb.set_ticklabels(tick_labels)

    # --- Save ---
    if filename:
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
