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
from matplotlib.ticker import ScalarFormatter
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import cartopy
import seaborn as sns

import numpy as np
from matplotlib.ticker import ScalarFormatter

def format_colorbar(cb, bounds, vmin=None, vmax=None):
    """
    Force-show ALL ticks at every bound.
    Uses scientific notation (powers of ten) only when magnitudes warrant it.
    Never shows ×10^0.
    """

    bounds = np.asarray(bounds, dtype=float)

    if vmin is None:
        vmin = float(np.nanmin(bounds))
    if vmax is None:
        vmax = float(np.nanmax(bounds))

    # Always show all bounds as ticks
    cb.set_ticks(bounds)

    # Decide whether scientific notation is needed based on magnitude
    max_abs = float(np.nanmax(np.abs([vmin, vmax])))
    use_sci = (max_abs >= 1e4) or (0 < max_abs < 1e-3)

    if use_sci:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 4))
        cb.ax.xaxis.set_major_formatter(fmt)

        cb.ax.figure.canvas.draw()
        off = cb.ax.xaxis.get_offset_text()
        txt = off.get_text().strip()
        if txt in ["×10⁰", "×10^0", r"$\times 10^{0}$", "x10^0", "x10⁰"]:
            off.set_text("")
    else:
        # Plain numeric labels
        rng = abs(vmax - vmin)
        if rng <= 10:
            labels = [f"{v:.2f}" for v in bounds]
        elif rng <= 100:
            labels = [f"{v:.1f}" for v in bounds]
        else:
            labels = [f"{v:.0f}" for v in bounds]

        cb.set_ticklabels(labels)
        cb.ax.xaxis.get_offset_text().set_text("")

    cb.ax.tick_params(axis='x', pad=6)


def plot_histogram(dataset, variable, time=None, bin_size=30, color='blue', plot_title=None, x_label=None, remove_outliers=False, log_transform=None, output_dir=None, filename=None):
    
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif isinstance(dataset, xr.Dataset):
        dataset = dataset
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")   
    
    # Ensure the specified variable is in the dataset
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
    
    # Handle time selection if provided
    if time is not None:
        dataset = dataset.sel(time=time, method='nearest').drop_vars('time')
        
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
    
    # --- Save ---
    if output_dir or filename:
        # Default filename if none provided
        filename = filename or "output_histogram.png"
        
        # Check for file extension
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"  # default to PNG if no extension provided
            filename = root + ext

        # Construct full save path
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

        # Save figure using detected or default format
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
    
    plt.show()
    
def plot_scatter(dataset, variable1, variable2, dataset2=None, time=None, color='blue', x_label=None, y_label=None, plot_title=None, remove_outliers=False, log_transform_1=None, log_transform_2=None, equation=False, output_dir=None, filename=None):

    # Load dataset1
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif not isinstance(dataset, xr.Dataset):
        raise TypeError("`dataset` must be an xarray.Dataset or a path to a NetCDF file.")

    # Load dataset2
    if dataset2 is None:
        dataset2 = dataset  # Use the same dataset if dataset2 is not provided
    elif isinstance(dataset2, (str, bytes, os.PathLike)):
        dataset2 = xr.open_dataset(dataset2)
    elif not isinstance(dataset2, xr.Dataset):
        raise TypeError("`dataset2` must be an xarray.Dataset, a path to a NetCDF file, or None.")

    # Check for required variables
    if variable1 not in dataset:
        raise ValueError(f"Variable '{variable1}' not found in the dataset.")

    if variable2 not in dataset2:
        raise ValueError(f"Variable '{variable2}' not found in the second dataset.")
    
    # Handle time selection if provided
    if time is not None:
        # Handle time selection for dataset1 if it has time dimension
        if 'time' in dataset.dims:
            dataset = dataset.sel(time=time, method='nearest').drop_vars('time')
                    # Handle time selection for dataset2 if it has time dimension
        if dataset2 is not None and 'time' in dataset2.dims:
            dataset2 = dataset2.sel(time=time, method='nearest').drop_vars('time')
    # Extract data
    data1 = dataset[variable1].values.flatten()
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
    
    # --- Save ---
    if output_dir or filename:
        # Default filename if none provided
        filename = filename or "output_scatter.png"
        
        # Check for file extension
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"  # default to PNG if no extension provided
            filename = root + ext

        # Construct full save path
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

        # Save figure using detected or default format
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))

    plt.show()

def plot_hexbin(dataset, variable1, variable2, dataset2=None, time=None, color='pink_r', grid_size=30, x_label=None, y_label=None, plot_title=None, remove_outliers=False, log_transform_1=None, log_transform_2=None, output_dir=None, filename=None):
    
    # Load dataset1
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif not isinstance(dataset, xr.Dataset):
        raise TypeError("`dataset` must be an xarray.Dataset or a path to a NetCDF file.")

    # Load dataset2
    if dataset2 is None:
        dataset2 = dataset  # Use the same dataset if dataset2 is not provided
    elif isinstance(dataset2, (str, bytes, os.PathLike)):
        dataset2 = xr.open_dataset(dataset2)
    elif not isinstance(dataset2, xr.Dataset):
        raise TypeError("`dataset2` must be an xarray.Dataset, a path to a NetCDF file, or None.")

    # Check for required variables
    if variable1 not in dataset:
        raise ValueError(f"Variable '{variable1}' not found in the dataset.")

    if variable2 not in dataset2:
        raise ValueError(f"Variable '{variable2}' not found in the second dataset.")

    # Handle time selection if provided
    if time is not None:
        # Handle time selection for dataset1 if it has time dimension
        if 'time' in dataset.dims:
            dataset = dataset.sel(time=time, method='nearest').drop_vars('time')
                    # Handle time selection for dataset2 if it has time dimension
        if dataset2 is not None and 'time' in dataset2.dims:
            dataset2 = dataset2.sel(time=time, method='nearest').drop_vars('time')

    # Extract data
    data1 = dataset[variable1].values.flatten()
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
    
    # --- Save ---
    if output_dir or filename:
        # Default filename if none provided
        filename = filename or "output_hexbin.png"
        
        # Check for file extension
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"  # default to PNG if no extension provided
            filename = root + ext

        # Construct full save path
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

        # Save figure using detected or default format
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
    
    plt.show()

def plot_time_series(dataset, variable, agg_function='sum', plot_type='both', color='blue', plot_label='Area Plot', x_label='Year', y_label='Value', plot_title='Time Series Plot', smoothing_window=None, output_dir=None, filename=None):
    
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        ds = xr.open_dataset(dataset)
    elif isinstance(dataset, xr.Dataset):
        ds = dataset
    else:
        raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")   
    
    # Ensure the specified variable is in the dataset
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
    
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
    
    # --- Save ---
    if output_dir or filename:
        # Default filename if none provided
        filename = filename or "output_time_series.png"
        
        # Check for file extension
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"  # default to PNG if no extension provided
            filename = root + ext

        # Construct full save path
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

        # Save figure using detected or default format
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
        
    plt.show()

# def plot_map(dataset, variable, time=None, color='hot_r', title='', label='', vmin=None, vmax=None, extend_min=False, extend_max=False, levels=10, out_bound=True, remove_ata=False, output_dir=None, filename=None, show=True):
    
#     # Load netcdf_file (either path or xarray.Dataset)
#     if isinstance(dataset, (str, bytes, os.PathLike)):
#         dataset = xr.open_dataset(dataset)
#     elif isinstance(dataset, xr.Dataset):
#         dataset = dataset
#     else:
#         raise TypeError("`netcdf_file` must be an xarray.Dataset or a path to a NetCDF file.")   
    
#     if time is not None:
#         dataset = dataset.sel(time=time, method='nearest').drop_vars('time')
#     # Ensure the specified variable is in the dataset
#     if variable not in dataset:
#         raise ValueError(f"Variable '{variable}' not found in the dataset.")

#     data = dataset[variable]
#     # Remove Antarctica if requested (e.g., keep only latitudes > -60°)
#     if remove_ata:
#         dataset = dataset.where(dataset['lat'] > -60, drop=True)
#         data = dataset[variable]


#     # Default vmin/vmax if not provided
#     if vmin is None:
#         vmin = data.min().item()
#     if vmax is None:
#         vmax = data.max().item()

#     # Data filtering based on rounding flags
#     extend = 'neither'
#     if extend_min and extend_max:
#         extend = 'both'
#     elif extend_min:
#         extend = 'min'
#         data = data.where(data <= vmax)
#     elif extend_max:
#         extend = 'max'
#         data = data.where(data >= vmin)
#     else:
#         data = data.where((data >= vmin) & (data <= vmax))
    
#     # Create levels and colormap
#     if isinstance(levels, list):
#         bounds = levels
#         num_levels = len(bounds) - 1
#     else:
#         step = (vmax - vmin) / levels
#         bounds = np.arange(vmin, vmax + step, step)
#         bounds = np.round(bounds, 2)
#         num_levels = len(bounds) - 1

#     # Updated colormap call (future-proof)
#     cmap_discrete = plt.get_cmap(color, num_levels)
    
#     # Color normalization
#     norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

#     # Plot
#     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(12, 6))
#     im = ax.pcolormesh(
#         dataset['lon'],
#         dataset['lat'],
#         data,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap_discrete,
#         norm=norm,
#     )

#     ax.coastlines(resolution='110m', color='gray', linewidth=1)
#     ax.add_feature(cfeature.LAND, color='white')
#     ax.set_title(title)
#     ax.spines['geo'].set_visible(out_bound)

#     # Colorbar
#     if remove_ata:
#         cax = fig.add_axes([0.27, 0.08, 0.5, 0.05])
#     else:
#         cax = fig.add_axes([0.27, 0.03, 0.5, 0.05])
        
#     cb = ColorbarBase(cax, cmap=cmap_discrete, norm=norm, orientation='horizontal', extend=extend)
#     cb.set_label(label)

#     # Format colorbar ticks based on data range
#     tick_values = bounds
#     if (vmax - vmin) <= 10:
#         tick_labels = [f"{val:.2f}" for val in tick_values]
#     else:
#         tick_labels = [f"{val:.0f}" for val in tick_values]

#     cb.set_ticks(tick_values)
#     cb.set_ticklabels(tick_labels)

#     # --- Save ---
#     if output_dir or filename:
#         # Default filename if none provided
#         filename = filename or "output_plot.png"
        
#         # Check for file extension
#         root, ext = os.path.splitext(filename)
#         if not ext:
#             ext = ".png"  # default to PNG if no extension provided
#             filename = root + ext

#         # Construct full save path
#         save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

#         # Save figure using detected or default format
#         plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))
    
#     if show:
#         plt.show()
        
#     return ax

def plot_map(dataset, variable, time=None, depth=None, color='hot_r', title='', label='', vmin=None, vmax=None, extend_min=None, extend_max=None, 
    levels=10, out_bound=True, remove_ata=False, output_dir=None, filename=None, show=True):
    # Load netcdf_file (either path or xarray.Dataset)
    if isinstance(dataset, (str, bytes, os.PathLike)):
        dataset = xr.open_dataset(dataset)
    elif isinstance(dataset, xr.Dataset):
        pass
    else:
        raise TypeError("`dataset` must be an xarray.Dataset or a path to a NetCDF file.")

    if time is not None:
        dataset = dataset.sel(time=time, method='nearest').drop_vars('time')
    if depth is not None:
        dataset = dataset.sel(depth=depth, method='nearest').drop_vars('depth')

    # Ensure the specified variable is in the dataset
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")

    # Optionally remove Antarctica
    if remove_ata:
        dataset = dataset.where(dataset['lat'] > -60, drop=True)

    data = dataset[variable]

    # Track whether user explicitly provided vmin/vmax
    vmin_given = vmin is not None
    vmax_given = vmax is not None

    # Default vmin/vmax if not provided
    if vmin is None:
        vmin = data.min().item()
    if vmax is None:
        vmax = data.max().item()

    # Auto-extend: if a limit was given, default to showing the arrow on that side
    # Unless extend_min/extend_max was explicitly provided True/False.
    if extend_min is None:
        extend_min = vmin_given
    if extend_max is None:
        extend_max = vmax_given

    # Prevent vmin > vmax
    if (vmin is not None) and (vmax is not None) and (vmin > vmax):
        vmin, vmax = vmax, vmin
        # If you want arrow meaning to follow the swapped limits, leave extend_* as-is.

    # Decide extend string for colorbar
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
    elif extend_max:
        extend = 'max'
    else:
        extend = 'neither'

    # Data filtering
    if not extend_min:
        data = data.where(data >= vmin)
    if not extend_max:
        data = data.where(data <= vmax)

    # Create levels and colormap
    if isinstance(levels, list):
        bounds = np.asarray(levels, dtype=float)
        num_levels = len(bounds) - 1
    else:
        if levels <= 0:
            raise ValueError("`levels` must be a positive int or a list of bounds.")
        step = (vmax - vmin) / levels
        bounds = np.arange(vmin, vmax + step, step)
        bounds = np.round(bounds, 2)
        num_levels = len(bounds) - 1

    cmap_discrete = plt.get_cmap(color, num_levels)
    norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N, clip=False)

    # Plot
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.Robinson()},
        figsize=(12, 6)
    )

    ax.pcolormesh(
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

    # Colorbar axes placement
    if remove_ata:
        cax = fig.add_axes([0.27, 0.08, 0.5, 0.05])
    else:
        cax = fig.add_axes([0.27, 0.03, 0.5, 0.05])

    cb = ColorbarBase(
        cax,
        cmap=cmap_discrete,
        norm=norm,
        orientation='horizontal',
        extend=extend
    )
    cb.set_label(label)

    # Format ticks
    tick_values = bounds
    if (vmax - vmin) <= 10:
        tick_labels = [f"{val:.2f}" for val in tick_values]
    else:
        tick_labels = [f"{val:.0f}" for val in tick_values]

    cb.set_ticks(tick_values)
    cb.set_ticklabels(tick_labels)
    format_colorbar(cb, bounds=bounds, vmin=vmin, vmax=vmax)

    # Save
    if output_dir or filename:
        filename = filename or "output_plot.png"
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
            filename = root + ext

        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))

    if show:
        plt.show()

    return ax



def plot_country(tabular_data, column, title="", label="", color='viridis', levels=10, output_dir=None, filename=None, remove_ata=False, out_bound=True, vmin=None, vmax=None, extend_min=None, extend_max=None, show=True):

    # Handle tabular_data input
    if isinstance(tabular_data, pd.DataFrame):
        dataframe = tabular_data
    elif isinstance(tabular_data, (str, bytes, os.PathLike)):
        try:
            dataframe = pd.read_csv(tabular_data, encoding='utf-8')
        except UnicodeDecodeError:
            dataframe = pd.read_csv(tabular_data, encoding='latin1')
    else:
        raise TypeError("`tabular_data` must be a pandas DataFrame or a path to a CSV file.")

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

    # Track whether caller provided limits
    vmin_given = vmin is not None
    vmax_given = vmax is not None

    # Default vmin/vmax from data if not provided
    if vmin is None:
        vmin = float(np.nanmin(data.to_numpy()))
    if vmax is None:
        vmax = float(np.nanmax(data.to_numpy()))

    # Auto-extend: enable arrow if limit was given, unless explicitly set to False
    if extend_min is None:
        extend_min = vmin_given
    if extend_max is None:
        extend_max = vmax_given

    # Prevent vmin > vmax
    if (vmin is not None) and (vmax is not None) and (vmin > vmax):
        vmin, vmax = vmax, vmin

    # Extend string for colorbar
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
    elif extend_max:
        extend = 'max'
    else:
        extend = 'neither'

    # Masking:
    # Clip only the sides you are NOT extending. Keep out-of-range values on extended sides.
    masked = data
    if not extend_min:
        masked = masked.where(masked >= vmin)
    if not extend_max:
        masked = masked.where(masked <= vmax)

    merged[column] = masked

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
    if remove_ata:
        # Restrict to 60°S
        ax.set_extent([-180, 180, -60, 90], crs=ccrs.Geodetic())
        
    cax = fig.add_axes([0.27, 0.08 if remove_ata else 0.03, 0.5, 0.05])
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
    format_colorbar(cb, bounds=bounds, vmin=vmin, vmax=vmax)

    # --- Save ---
    if output_dir or filename:
        # Default filename if none provided
        filename = filename or "output_country_plot.png"
        
        # Check if the filename includes an extension
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"  # default to PNG if no extension provided
            filename = root + ext

        # Construct full save path
        save_path = os.path.join(output_dir if output_dir else os.getcwd(), filename)

        # Save the plot with appropriate format
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format=ext.lstrip('.'))

    if show:
        plt.show()

    return ax
