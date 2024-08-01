import os
import sys
import unittest
import warnings
import xarray as xr

# Identify the base directory of the current script
base_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate one directory back
parent_directory = os.path.dirname(base_directory)

# Construct the path to the sesame_interface directory
sesame_path = os.path.join(parent_directory, 'src')

# Append this path to sys.path
sys.path.append(sesame_path)

import sesametoolbox as st
import utils

# Suppress specific RuntimeWarnings globally
warnings.filterwarnings("ignore", message="numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject")
warnings.simplefilter("ignore", category=RuntimeWarning)

class TestSesameToolbox(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.")
    
    def setUp(self):
        # Base path relative to the script's directory
        self.base_path = os.path.dirname(__file__)

    def test_table_2_grid(self):
        # Test function_table_2_grid
        print("Checking table_2_grid...")

        # Define paths and parameters
        netcdf_file_path = os.path.join(self.base_path, "data", "line_2_grid.nc")
        csv_file_path = os.path.join(self.base_path, "data", "railtrack_material.csv")
        variable_name = 'railtract_steel'
        long_name = 'Railtrack Steel Mass'
        units = 'g m-2'
        source = ("UNECE — United Nations Economic Commission of Europe (2021) Electronic dataset, downloaded 16 November 2021. "
                    "https://w3.unece.org/PXWeb2015/pxweb/en/STAT/STAT__40-TRTRANS__11-TRINFRA/ZZZ_en_TRRailInfra1_r.px/.\n"
                    "CIA — Central Intelligence Agency (2021a) “The World Factbook — Railways”. Electronic dataset, visited 2 December 2021. "
                    "https://www.cia.gov/the-world-factbook/field/railways/. \n"
                    "World Bank (2021). “Rail lines (total route-km)”, electronic dataset, downloaded 16 November 2021. "
                    "https://data.worldbank.org/indicator/IS.RRS.TOTL.KM?view=map&year=2018.")
        
        # Load the expected result
        expected_result = xr.load_dataset(os.path.join(self.base_path, "data", "table_2_grid.nc"))

        # Run the function
        result = st.table_2_grid(netcdf_variable="railway_length", tabular_column="steel", netcdf_file_path=netcdf_file_path, 
                                csv_file_path=csv_file_path, variable_name=variable_name, long_name=long_name, units=units, 
                                source=source, value_per_area="yes", verbose="yes")
        
        # Assert the result matches the expected output
        xr.testing.assert_identical(result, expected_result)
        
        # Measure and print execution time
        print(f"table_2_grid test passed.")

    '''

    def test_grid_2_grid(self):
        print("Checking grid_2_grid...")
        
        # Test function_four
        netcdf_path = os.path.join(self.base_path, "data", "MERRA2_200.tavgM_2d_aer_Nx.200001.nc4")
        variable_name = "black_carbon"
        netcdf_variable= "BCANGSTR"
        time = "2000-01-01"
        fold_function = "mean"
        long_name = "Black Carbon Angstrom parameter [470-870 nm]"
        source = "Global Modeling and Assimilation Office (GMAO) (2015), MERRA-2 tavgM_2d_aer_Nx: 2d,Monthly mean,Time-averaged,Single-Level,Assimilation,Aerosol Diagnostics V5.12.4, Greenbelt, MD, USA, Goddard Earth Sciences Data and Information Services Center (GES DISC), 10.5067/FH9A0MLJPC7N"
        units = "1"
        expected_result = xr.load_dataset(os.path.join(self.base_path, "data", "grid_2_grid.nc"))
        
        result = st.grid_2_grid(raster_path=netcdf_path, fold_function=fold_function, variable_name=variable_name, long_name=long_name, 
                    units=units, source=source, time=time, cell_size=1, netcdf_variable=netcdf_variable, verbose=True)
        
        xr.testing.assert_identical(result, expected_result)
        print("grid_2_grid test passed.")

    
    def test_point_2_grid(self):
        # Test function_one
        print("Checking point_2_grid...")
        
        input_shapefile = os.path.join(self.base_path, "data", "airports.shp")
        variable_name ="airplanes"
        long_name="Airplanes Count"
        units="airport/grid-cell"
        source="CIA — Central Intelligence Agency (2021) “The World Factbook — National Air Transport System”. Electronic dataset, visited 27 November 2021. https://www.cia.gov/the-world-factbook/field/national-air-transport-system/"
        
        expected_result = xr.load_dataset(os.path.join(self.base_path, "data", 'point_2_grid.nc'))
        
        result = st.point_2_grid(variable_name=variable_name, long_name=long_name, units=units, source=source, 
                                cell_size=1, shapefile_path=input_shapefile, verbose=True)
        
        xr.testing.assert_identical(result, expected_result)
        print("point_2_grid test passed.")
        
    
    def test_line_2_grid(self):
        print("Checking line_2_grid...")

        # Define paths and parameters
        input_shapefile = os.path.join(self.base_path, "data", "Global_Railways_WFP.shp")
        variable_name = 'railway_length'
        long_name = 'Total Railway Length in km'
        source = 'Global Railways (WFP-World Food Programme SDI-T - Logistics Database)'

        expected_result = xr.load_dataset(os.path.join(self.base_path, "data", "line_2_grid.nc"))


        result = st.line_2_grid(variable_name=variable_name, long_name=long_name, units="meter/grid-cell", source=source,
                        cell_size=1, fold_field=None, fold_function="sum", attr_field=None, shapefile_path=input_shapefile, 
                        verbose="yes")

        # Assert all variables are close with increased tolerance
        for varname in expected_result.variables:
            xr.testing.assert_allclose(result[varname], expected_result[varname], rtol=1e-2, atol=1e-2)

        print("line_2_grid test passed.")
    
    def test_poly_2_grid(self):
        # Test function_three
        print("Checking poly_2_grid...")
        input_shapefile = os.path.join(self.base_path, "data", "glim_wgs84_0point5deg.shp")
        source = "Hartmann, J., Moosdorf, N., 2012. The new global lithological map database GLiM: A representation of rock properties at the Earth surface. Geochemistry, Geophysics, Geosystems, 13. DOI: 10.1029/2012GC004370"
        units = "fraction"
        expected_result = xr.load_dataset(os.path.join(self.base_path, "data",'poly_2_grid.nc'))
        result = st.poly_2_grid(shapefile_path=input_shapefile, units=units, source=source, cell_size=1, 
                                attr_field="Short_Name", fraction="yes", verbose=True)
        
        # Assert all variables are close with increased tolerance
        for varname in expected_result.variables:
            xr.testing.assert_allclose(result[varname], expected_result[varname], rtol=1e-2, atol=1e-2)

        print("poly_2_grid test passed.")
    '''
        

# Identify the base directory of the current script
base_directory = os.path.dirname(os.path.abspath(__file__))

def delete_temporary_folder():
    temp_directory = os.path.join(base_directory, "temp")
    utils.delete_temporary_folder(temp_directory)

if __name__ == '__main__':
    try:
        # Run tests
        unittest.main(verbosity=2)
    finally:
        # Delete temporary folder after tests complete
        delete_temporary_folder()
    
