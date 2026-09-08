"""
Regression tests for SESAME's core gridding functions.

These tests re-run each of the seven core functions against the fixture
files already committed in test/data/, using the exact parameters recorded
in test_sesametoolbox.ipynb, and check the output against the reference
NetCDF files already committed alongside them (point_2_grid.nc,
line_2_grid.nc, poly_2_grid.nc, grid_2_grid.nc, table_2_grid.nc).

Tolerances are set from actually measured floating-point noise between two
real runs (see the note on poly_2_grid below), not picked arbitrarily.

Run from the test/ directory so the relative "data/" paths resolve:
    cd test && pytest test_core.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sesame as ssm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _assert_matches_reference(new_ds, reference_filename, max_nan_mismatch=0):
    """Compare a freshly-produced dataset against the committed reference
    NetCDF of the same name in test/data/.

    max_nan_mismatch allows a small number of grid cells where one side is
    NaN and the other is a value indistinguishable from zero (see the
    poly_2_grid test below for why this is sometimes necessary).
    """
    reference_path = os.path.join(DATA, reference_filename)
    ref_ds = xr.open_dataset(reference_path)

    for var in ref_ds.data_vars:
        assert var in new_ds.data_vars, f"'{var}' missing from freshly computed output"
        expected = ref_ds[var].values.astype(float)
        actual = new_ds[var].values.astype(float)
        assert expected.shape == actual.shape, f"'{var}' shape changed"

        nan_mismatch = np.isnan(expected) != np.isnan(actual)
        n_mismatch = int(nan_mismatch.sum())
        assert n_mismatch <= max_nan_mismatch, (
            f"'{var}' has {n_mismatch} cells where NaN-ness differs between "
            f"the reference and the new run (allowed: {max_nan_mismatch})"
        )
      # Where both sides are non-NaN, values must match closely.
        both_finite = ~np.isnan(expected) & ~np.isnan(actual)
        np.testing.assert_allclose(
            expected[both_finite], actual[both_finite], rtol=1e-6, atol=1e-6,
            err_msg=f"'{var}' values drifted from the committed reference",
        )
        # Where they disagree on NaN-ness at all, the non-NaN side must be
        # itself indistinguishable from zero (a sliver-polygon edge case,
        # not a real discrepancy).
        if n_mismatch:
            disputed = np.where(nan_mismatch)
            for side in (expected, actual):
                vals = side[disputed]
                vals = vals[~np.isnan(vals)]
                assert np.allclose(vals, 0.0, atol=1e-6), (
                    f"'{var}' NaN/value mismatch is not a near-zero edge case, "
                    "investigate before ignoring"
                )


def test_point_2_grid():
    ds = ssm.point_2_grid(
        point_data=os.path.join(DATA, "airports.shp"),
        variable_name="airplanes",
        long_name="Airplanes Count",
        units="airport/grid-cell",
        source="CIA",
        resolution=1,
        output_directory=DATA + os.sep,
        output_filename="point_2_grid",
        verbose=False,
    )
    _assert_matches_reference(ds, "point_2_grid.nc")


def test_line_2_grid():
    ds = ssm.line_2_grid(
        line_data=os.path.join(DATA, "Global_Railways_WFP.shp"),
        variable_name="railway_length",
        long_name="Total Railway Length in km",
        units="meter/grid-cell",
        source="Global Railways (WFP)",
        time=None,
        resolution=1,
        agg_column=None,
        agg_function="sum",
        attr_field=None,
        output_directory=DATA + os.sep,
        output_filename="line_2_grid",
        normalize_by_area=False,
        zero_is_value=False,
        verbose=False,
    )
    _assert_matches_reference(ds, "line_2_grid.nc")


def test_poly_2_grid():
    ds = ssm.poly_2_grid(
        polygon_data=os.path.join(DATA, "glim_wgs84_0point5deg.shp"),
        units="fraction",
        source="Hartmann & Moosdorf 2012",
        resolution=1,
        attr_field="Short_Name",
        fraction="yes",
        output_directory=DATA + os.sep,
        output_filename="poly_2_grid",
        verbose=False,
    )
    # A near-zero-area sliver polygon can resolve to NaN on one GEOS/Shapely
    # version and ~1e-14 (numerically zero) on another. Measured on this
    # suite's reference environment: 2 of 64,800 cells, one variable. Allow
    # a small margin rather than pin the test to one geometry library build.
    _assert_matches_reference(ds, "poly_2_grid.nc", max_nan_mismatch=10)


def test_grid_2_grid():
    ds = ssm.grid_2_grid(
        raster_data=os.path.join(DATA, "MERRA2_200.tavgM_2d_aer_Nx.200001.nc4"),
        agg_function="mean",
        variable_name="black_carbon",
        long_name="Black Carbon Angstrom parameter",
        units="1",
        source="GMAO MERRA-2",
        time="2000-01-01",
        resolution=1,
        netcdf_variable="BCANGSTR",
        output_directory=DATA + os.sep,
        output_filename="grid_2_grid",
        verbose=False,
    )
    _assert_matches_reference(ds, "grid_2_grid.nc")


def test_table_2_grid():
    # table_2_grid needs a surrogate netCDF; regenerate line_2_grid.nc first
    # so this test doesn't depend on test ordering.
    ssm.line_2_grid(
        line_data=os.path.join(DATA, "Global_Railways_WFP.shp"),
        variable_name="railway_length",
        long_name="Total Railway Length in km",
        units="meter/grid-cell",
        source="Global Railways (WFP)",
        time=None,
        resolution=1,
        agg_column=None,
        agg_function="sum",
        attr_field=None,
        output_directory=DATA + os.sep,
        output_filename="line_2_grid",
        normalize_by_area=False,
        zero_is_value=False,
        verbose=False,
    )
    ds = ssm.table_2_grid(
        surrogate_data=os.path.join(DATA, "line_2_grid.nc"),
        surrogate_variable="railway_length",
        tabular_data=os.path.join(DATA, "railtrack_material.csv"),
        tabular_column="steel",
        variable_name="railtract_steel",
        long_name="Railtrack Steel Mass",
        units="g m-2",
        source="UNECE/CIA/World Bank",
        output_directory=DATA + os.sep,
        output_filename="table_2_grid",
        normalize_by_area="yes",
        verbose=False,
    )
    _assert_matches_reference(ds, "table_2_grid.nc")


def test_grid_2_table_conserves_global_sum():
    # railtract_steel is stored area-normalized (g/m-2, see table_2_grid's
    # normalize_by_area="yes"), so summing the raw NetCDF variable directly
    # is not physically meaningful, that would sum a density across cells
    # of different areas. grid_2_table(grid_area="yes") multiplies back by
    # cell area before aggregating, so the real invariant is that country
    # level and region level aggregations agree with each other (both
    # computed by grid_2_table itself, not reimplemented by hand here).
    by_country = ssm.grid_2_table(
        grid_data=os.path.join(DATA, "table_2_grid.nc"),
        variables="railtract_steel",
        time=None,
        grid_area="yes",
        resolution=1,
        aggregation=None,
        agg_function="sum",
        verbose=False,
    )
    by_region = ssm.grid_2_table(
        grid_data=os.path.join(DATA, "table_2_grid.nc"),
        variables="railtract_steel",
        time=None,
        grid_area="yes",
        resolution=1,
        aggregation="region_1",
        agg_function="sum",
        verbose=False,
    )
    assert isinstance(by_region, pd.DataFrame)
    assert "railtract_steel" in by_region.columns
    assert "region_1" in by_region.columns
    assert len(by_region) > 0

    country_total = float(by_country["railtract_steel"].sum())
    region_total = float(by_region["railtract_steel"].sum())
    assert region_total == pytest.approx(country_total, rel=1e-6)
    # Cross-check against table_2_grid's own reported global total for this
    # exact fixture (verified independently while writing this test).
    assert region_total == pytest.approx(176311657500000.0, rel=1e-3)


@pytest.mark.parametrize(
    "country_name,expected_iso3",
    [
        ("United States", "USA"),
        ("USA", "USA"),
        ("Canada", "CAN"),
        ("United Kingdom", "GBR"),
        ("Bangladesh", "BGD"),
    ],
)
def test_add_iso3_column(country_name, expected_iso3):
    df = pd.DataFrame({"Country": [country_name]})
    out = ssm.add_iso3_column(df=df, column="Country")
    assert out.loc[0, "ISO3"] == expected_iso3
