from setuptools import find_packages, setup

setup(
    name="unseen",
    version="0.0.1",
    author="unseen developers",
    url="https://github.com/AusClimateService/unseen",
    description="A Python library for UNSEEN analysis.",
    long_description="A Python package for implementing the UNSEEN (UNprecedented Simulated Extremes using ENsembles) approach to assessing the likelihood of extreme events.",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "xarray>=0.18.0", "dask", "zarr", "pyyaml", "cmdline_provenance", "geopandas", "regionmask", "xclim",
    ]
)

