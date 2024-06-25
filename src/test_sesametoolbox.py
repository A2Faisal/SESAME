import os
import unittest
import warnings
import xarray as xr
import sesametoolbox

# Suppress specific RuntimeWarnings globally
warnings.filterwarnings("ignore", message="numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject")
warnings.simplefilter("ignore", category=RuntimeWarning)

class TestSesameToolbox(unittest.TestCase):

    def setUp(self):
        # Base path relative to the script's directory
        self.base_path = os.path.dirname(__file__)
        

    def test_table_2_grid(self):
        # Test function_table_2_grid
        print("Checking table_2_grid...")

        # Define paths and parameters
        netcdf_file_path = os.path.join(self.base_path, "test", "data", "line_2_grid.nc")
        csv_file_path = os.path.join(self.base_path, "test", "data", "railtrack_material.csv")
        short_name = 'railtract_steel'
        long_name = 'Railtrack Steel Mass'
        units = 'g m-2'
        source = ("UNECE — United Nations Economic Commission of Europe (2021) Electronic dataset, downloaded 16 November 2021. "
                  "https://w3.unece.org/PXWeb2015/pxweb/en/STAT/STAT__40-TRTRANS__11-TRINFRA/ZZZ_en_TRRailInfra1_r.px/.\n"
                  "CIA — Central Intelligence Agency (2021a) “The World Factbook — Railways”. Electronic dataset, visited 2 December 2021. "
                  "https://www.cia.gov/the-world-factbook/field/railways/. \n"
                  "World Bank (2021). “Rail lines (total route-km)”, electronic dataset, downloaded 16 November 2021. "
                  "https://data.worldbank.org/indicator/IS.RRS.TOTL.KM?view=map&year=2018.")
        
        # Load the expected result
        expected_result = xr.load_dataset(os.path.join(self.base_path, "test", "data", "table_2_grid.nc"))

        # Run the function
        result = sesametoolbox.table_2_grid(
            netcdf_variable="railway_length", 
            tabular_column="steel", 
            netcdf_file_path=netcdf_file_path, 
            csv_file_path=csv_file_path, 
            short_name=short_name, 
            long_name=long_name, 
            units=units, 
            source=source, 
            value_per_sqm="yes"
        )
        
        # Assert the result matches the expected output
        xr.testing.assert_identical(result, expected_result)
        
        # Measure and print execution time
        print(f"table_2_grid test passed.")
        
    def test_grid_2_grid(self):
        print("Checking grid_2_grid...")
        # Test function_four
        tif_path = os.path.join(self.base_path, "test", "data", "NPP_2001.tif")
        source = (
            "Running, S., Zhao, M. (2021). MODIS/Terra Net Primary Production Gap-Filled Yearly L4 Global 500m SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. "
            "Accessed 2024-05-17 from https://doi.org/10.5067/MODIS/MOD17A3HGF.061"
        )

        fold_function = "MEAN"
        short_name = 'npp_terrestrial'
        long_name = 'Terrestrial Net Primary Production (NPP)'
        units = "kg C m-2 y-1"
        time = "2001-01-01"
        expected_result = xr.load_dataset(os.path.join(self.base_path, "test", "data", "grid_2_grid.nc"))
        result = sesametoolbox.grid_2_grid(
            raster_path=tif_path, 
            fold_function=fold_function, 
            short_name=short_name, 
            long_name=long_name, 
            units=units, 
            source=source, 
            time=time, 
            cell_size=1
        )
        xr.testing.assert_identical(result, expected_result)
        print("grid_2_grid test passed.")

        
    
    def test_point_2_grid(self):
        # Test function_one
        print("Checking point_2_grid...")
        short_name="airplanes"
        long_name="Airplanes Count"
        units="airport/grid-cell"
        source="CIA — Central Intelligence Agency (2021) “The World Factbook — National Air Transport System”. Electronic dataset, visited 27 November 2021. https://www.cia.gov/the-world-factbook/field/national-air-transport-system/"
        
        input_shapefile = os.path.join(self.base_path, "test", "data", "airports.shp")
        expected_result = xr.load_dataset(os.path.join(self.base_path, "test", "data", 'point_2_grid.nc'))
        
        result = sesametoolbox.point_2_grid(
                shapefile_path=input_shapefile, 
                short_name=short_name, 
                long_name=long_name, 
                units=units, 
                source=source, 
                attr_field="type"
        )
        xr.testing.assert_identical(result, expected_result)
        print("point_2_grid test passed.")
        
    
    def test_line_2_grid(self):
        print("Checking line_2_grid...")

        # Define paths and parameters
        input_shapefile = os.path.join(self.base_path, "test", "data", "Global_Railways_WFP.shp")
        short_name = 'railway_length'
        long_name = 'Total Railway Length in km'
        source = 'Global Railways (WFP-World Food Programme SDI-T - Logistics Database)'

        expected_result = xr.load_dataset(os.path.join(self.base_path, "test", "data", "line_2_grid.nc"))

        # Run the function
        result = sesametoolbox.line_2_grid(
            shapefile_path=input_shapefile,
            source=source,
            cell_size=1,
            units='km/grid-cell',
            short_name=short_name,
            long_name=long_name
        )

        # Assert all variables are close with increased tolerance
        for varname in expected_result.variables:
            xr.testing.assert_allclose(result[varname], expected_result[varname], rtol=1e-2, atol=1e-2)

        print("line_2_grid test passed.")
        
        
    def test_poly_2_grid(self):
        # Test function_three
        print("Checking poly_2_grid...")
        input_shapefile = os.path.join(self.base_path, "test", "data", "glim_wgs84_0point5deg.shp")
        source = "Hartmann, J., Moosdorf, N., 2012. The new global lithological map database GLiM: A representation of rock properties at the Earth surface. Geochemistry, Geophysics, Geosystems, 13. DOI: 10.1029/2012GC004370"
        units = "fraction"
        expected_result = xr.load_dataset(os.path.join(self.base_path, "test", "data",'poly_2_grid.nc'))
        result = sesametoolbox.poly_2_grid(
                    shapefile_path=input_shapefile, 
                    source=source, 
                    cell_size=1, 
                    units=units, 
                    attr_field="Short_Name", 
                    fraction="Yes")
        xr.testing.assert_identical(result, expected_result)
        print("poly_2_grid test passed.")    
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
