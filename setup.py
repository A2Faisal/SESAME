from setuptools import setup, find_packages

setup(
    name="sesame-iesd",
    version="0.1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="A simple Python package for Sesame toolbox",
    author="Abdullah Al Faisal,Maxwell Kaye",
    author_email="abdullah.al.faisal@mail.mcgill.ca, maxwell.kaye@mail.mcgill.ca",
    license="MIT",
    install_requires=[
		"geopandas>=1.0.1",
		"xarray>=2024.6.0",
		"h5netcdf>=1.3.0",
		"rasterio>=1.3.10",
		"matplotlib>=3.9.1",
		"seaborn>=0.13.2",
		"scipy>=1.14.0",
		"cartopy==0.23.0"
	],
    extras_require={
        "dev": ["pytest"],
    },
    test_suite="tests",
    include_package_data=True,
)

