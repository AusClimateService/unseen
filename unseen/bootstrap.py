"""Utilities for repeated random sampling"""

import pdb

import numpy as np
import xarray as xr
import dask


def random_resample(*args, samples,
                    function=None, function_kwargs=None, bundle_args=True,
                    replace=True):
    """Randomly resample from provided xarray args and return the results of the subsampled dataset passed through \
        a provided function
                
        Parameters
        ----------
        *args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}. The first \
            object in args must contain all dimensions listed in samples, but subsequent objects need not.
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        replace : boolean, optional
            Whether the sample is with or without replacement
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
    """
    samples_spec = samples.copy() # copy because use pop below
    args_sub = [obj.copy() for obj in args]
    dim_block_1 = [d for d, s in samples_spec.items() if s[1] == 1]

    # Do all dimensions with block_size = 1 together
    samples_block_1 = { dim: samples_spec.pop(dim) for dim in dim_block_1 }
    random_samples = {dim: 
                      np.random.choice(
                          len(args_sub[0][dim]),
                          size=n,
                          replace=replace)
                      for dim, (n, _) in samples_block_1.items()}
    args_sub = [obj.isel(
        {dim: random_samples[dim] 
         for dim in (set(random_samples.keys()) & set(obj.dims))}) for obj in args_sub]

    # Do any remaining dimensions
    for dim, (n, block_size) in samples_spec.items():
        n_blocks = int(n / block_size)
        random_samples = [slice(x,x+block_size) 
                          for x in np.random.choice(
                              len(args_sub[0][dim])-block_size+1, 
                              size=n_blocks,
                              replace=replace)]
        args_sub = [xr.concat([obj.isel({dim: random_sample}) 
                               for random_sample in random_samples],
                              dim=dim) 
                       if dim in obj.dims else obj 
                       for obj in args_sub]

    if function:
        if bundle_args:
            if function_kwargs is not None:
                res = function(*args_sub, **function_kwargs)
            else:
                res = function(*args_sub)
        else:
            if function_kwargs is not None:
                res = tuple([function(obj, **function_kwargs) for obj in args_sub])
            else:
                res = tuple([function(obj) for obj in args_sub])
    else:
        res = tuple(args_sub,)

    if isinstance(res, tuple) & len(res) == 1:
        return res[0]
    else:
        return res


def n_random_resamples(*args, samples, n_repeats, 
                       function=None, function_kwargs=None, bundle_args=True, 
                       replace=True, with_dask=True):
    """
        Repeatedly randomly resample from provided xarray objects and return the results of the subsampled dataset passed \
        through a provided function
                
        Parameters
        ----------
        args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}
        n_repeats : int
            Number of times to repeat the resampling process
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        replace : boolean, optional
            Whether the sample is with or without replacement
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        with_dask : boolean, optional
            If True, use dask to parallelize across n_repeats using dask.delayed
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
    """

    if with_dask & (n_repeats > 500):
        n_args = itertools.repeat(args[0], times=n_repeats)
        b = db.from_sequence(n_args, npartitions=100)
        rs_list = b.map(random_resample, *(args[1:]), 
                        **{'samples':samples, 'function':function, 
                           'function_kwargs':function_kwargs, 'replace':replace}).compute()
    else:              
        resample_ = dask.delayed(random_resample) if with_dask else random_resample
        rs_list = [resample_(*args,
                             samples=samples,
                             function=function,
                             function_kwargs=function_kwargs,
                             bundle_args=bundle_args,
                             replace=replace) for _ in range(n_repeats)] 
        if with_dask:
            rs_list = dask.compute(rs_list)[0]
            
    if all(isinstance(r, tuple) for r in rs_list):
        return tuple([xr.concat([r.unify_chunks() for r in rs], dim='k') for rs in zip(*rs_list)])
    else:
        return xr.concat([r.unify_chunks() for r in rs_list], dim='k')


