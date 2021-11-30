"""Setup dask scheduling client"""

import pdb

import yaml
import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster, SLURMCluster


def launch_client(config_file):
    """Launch a dask client."""

    with open(config_file, "r") as reader:
        config_dict = yaml.load(reader, Loader=yaml.FullLoader)

    if "temporary_directory" in config_dict:
        dask.config.set(temporary_directory=config_dict["temporary_directory"])

    if "LocalCluster" in config_dict:
        cluster = LocalCluster(**config_dict["LocalCluster"])
    elif "PBSCluster" in config_dict:
        cluster = PBSCluster(**config_dict["PBSCluster"])
    elif "SLURMCluster" in config_dict:
        cluster = SLURMCluster(**config_dict["SLURMCluster"])
    else:
        raise ValueError("No recognised clusters in dask config file")

    client = Client(cluster)
    print("Watch progress at http://localhost:8787/status")

    return client
