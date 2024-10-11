from setuptools import find_packages, setup
import versioneer

setup(
    name="unseen",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="unseen developers",
    url="https://github.com/AusClimateService/unseen",
    description="A Python library for UNSEEN analysis.",
    long_description="A Python library for performing UNSEEN analysis for assessing the likelihood of extreme events.",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "xarray",
        "dask",
    ],
    entry_points={
        "console_scripts": [
            "fileio = unseen.fileio:_main",
            "independence = unseen.independence:_main",
            "similarity = unseen.similarity:_main",
            "bias_correction = unseen.bias_correction:_main",
            "stability = unseen.stability:_main",
            "moments = unseen.moments:_main",
            "eva = unseen.eva:_main",
        ]
    },
)
