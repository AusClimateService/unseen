"""Setup dask scheduling"""

from dask.distributed import Client, LocalCluster


def local():
    """Launch a local dask client

    Watch progress at http://localhost:8787/status
    """

    cluster = LocalCluster()
    client = Client(cluster)


