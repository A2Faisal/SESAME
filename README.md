# SESAME Project

## About

The **SESAME project** aims to unify key human and non-human (Earth system) datasets into a shared spatially-gridded format. These datasets are often siloed in incompatible formats; SESAME bridges that gap to accelerate scientific discovery and interdisciplinary modeling.

---

## 📑 Table of Contents

- [Goals](#goals)
- [Example Tools](#example-tools)
- [Human-Earth Atlas](#human-earth-atlas)
- [Software Documentation](#software-documentation)
- [Installation Instructions](#installation-instructions)
- [Contact](#contact)

---

## Goals

- Unify human and non-human system datasets in a standardized spatially-gridded structure.
- Improve data discoverability and interoperability for research and modeling.
- Support interdisciplinary science through accessible, integrated datasets.

---

## Example Tools

- **`point_2_grid`**  
  Maps point data onto standardized global grids. Supports counting points per cell, summing or averaging associated values, or grouping by class to generate multi-variable datasets.

- **`line_2_grid`**  
  Maps line data onto global grids by calculating the length of each line segment within grid cells. Supports aggregation methods such as mean, max, or standard deviation through spatial intersections.

- **`poly_2_grid`**  
  Handles polygon data by computing the fraction or area of each polygon that overlaps with each grid cell. Supports combining multiple polygon types into a multi-variable NetCDF output.

- **`grid_2_grid`**  
  Converts raster data to a new grid resolution. Ensures global coverage, checks projections, fills in missing cells with NaNs, and supports aggregation methods like sum, mean, max, min, or standard deviation.

- **`table_2_grid`**  
  Converts jurisdiction-level tabular data into standardized grids using surrogate variables. Accounts for boundary changes over time for accurate spatial representation.

- **`add_iso3_column`**  
  Standardizes country names by converting them to ISO3 codes. This function is a prerequisite for running `table_2_grid`.

- **`grid_2_table`**  
  Reverses the gridding process by aggregating gridded data into summary tables based on predefined regions or countries.

- **Built-in plotting functions**  
  SESAME also includes tools for quick visualization and mapping, such as `plot_histogram`, `plot_scatter`, `plot_time_series`, and `plot_hexbin` for charts, and `plot_map`, `plot_country` for gridded or choropleth maps. These functions help generate both exploratory plots and publication-ready outputs with minimal setup.

---

## Human-Earth Atlas

**Atlas Data Access**:

Faisal, A. A., Kaye, M., Ahmed, M. & Galbraith, E. _The SESAME Human-Earth Atlas_. figshare [https://doi.org/10.6084/m9.figshare.28432499](https://doi.org/10.6084/m9.figshare.28432499) (2025).  

**Paper**

Faisal, A.A., Kaye, M., Ahmed, M. et al. (2025)._The SESAME Human-Earth Atlas_. *Scientific Data*, 12, 775. [https://doi.org/10.1038/s41597-025-05087-5](https://doi.org/10.1038/s41597-025-05087-5)

---

## Software Documentation

Detailed setup, usage instructions, advanced features, and testing procedures are included in the official software manual:

📄 [**SESAME Software Manual (v1.1)**](https://a2faisal.github.io/SESAME/)

---

## Installation Instructions

If you're using **conda** to manage your Python environment:

### Step 1: Create and activate the environment

```bash
# create a new conda environment
conda create -n sesame_env
# activate the environment
conda activate sesame_env
# install pip
conda install pip
# install SESAME from testPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sesame-iesd==0.1.0
```
### Step 2: Add this as a Python code snippet:
```bash
import sesame as ssm 
```

### If you are using Windows and encounter the error: 
"ERROR: Failed building wheel for cartopy," please follow these steps:

1. Download and install the Microsoft C++ Build Tools from the official website:  
   [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2. During installation, make sure to select the **"Desktop development with C++"** workload.  
   This is required to build packages like cartopy that depend on C++ extensions.

![Visual Studio Build Tools Installation](https://github.com/A2Faisal/SESAME/blob/main/docs/images/vs_build_tools.png)


## Contact
For questions or inquiries about the SESAME project, please contact [abdullah-al.faisal@mail.mcgill.ca](mailto:abdullah-al.faisal@mail.mcgill.ca) or [maxwell.kaye@mail.mcgill.ca](mailto:maxwell.kaye@mail.mcgill.ca).
