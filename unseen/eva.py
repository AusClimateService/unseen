"""Extreme value analysis functions."""

import argparse
from lmoments3 import distr
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.dates import date2num
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme, goodness_of_fit
from scipy.stats.distributions import chi2
import warnings
from xarray import apply_ufunc, DataArray
import xclim.indices.stats as xcstats

from . import fileio
from . import general_utils
from . import time_utils


def event_in_context(data, threshold, direction):
    """Put an event in context.

    Parameters
    ----------
    data : numpy ndarray
        Population data
    threshold : float
        Event threshold
    direction : {'above', 'below'}
        Provide statistics for above or below threshold

    Returns
    -------
    n_events : int
        Number of events in population
    n_population : int
        Size of population
    return_period : float
        Return period for event
    percentile : float
        Event percentile relative to population (%)
    """

    n_population = len(data)
    if direction == "below":
        n_events = np.sum(data < threshold)
    elif direction == "above":
        n_events = np.sum(data > threshold)
    else:
        raise ValueError("""direction must be 'below' or 'above'""")
    percentile = (np.sum(data < threshold) / n_population) * 100
    return_period = n_population / n_events

    return n_events, n_population, return_period, percentile


def fit_gev(
    data,
    core_dim="time",
    stationary=True,
    covariate=None,
    fitstart="LMM",
    loc1=0,
    scale1=0,
    assert_good_fit=False,
    pick_best_model=False,
    alpha=0.05,
    method="Nelder-Mead",
):
    """Estimate stationary or nonstationary GEV distribution parameters.

    Parameters
    ----------
    data : array_like
        Data to use in estimating the distribution parameters
    core_dim : str, default "time"
        Name of time/sample dimension in `data` and `covariate`
    stationary : bool, default True
        Fit as a stationary GEV using `fit_stationary_gev`
    covariate : array_like, optional
        A nonstationary covariate array with the same `core_dim` as `data`
    fitstart : {array-like, 'LMM', 'MM', 'scipy', 'scipy_fitstart',
    'scipy_subset', 'xclim_fitstart', 'xclim'}, default 'scipy_fitstart'
        Initial guess method/estimate of the shape, loc and scale parameters
    loc1, scale1 : float or None, default 0
        Initial guess of trend parameters. If None, the trend is fixed at zero
    assert_good_fit : bool, default False
        Stationary parameters must pass goodness of fit test at `alpha` level.
        Attempt a retry and return NaNs if the test fails again.
    pick_best_model : {False, 'lrt', 'aic', 'bic'}, default False
        Method to test relative fit of stationary and nonstationary models.
        Do not use if you don't want nonstationary parameters. The output will
        have GEV 5 parameters even if stationary is True.
    alpha : float, default 0.05
        Fit test p-value threshold for stationary fit (relative/goodness of fit)
    method : {'Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell',
    'trust-constr', 'COBYLA'}, default 'Nelder-Mead'
        Optimization method for nonstationary fit

    Returns
    -------
    dparams : xarray.DataArray
        The GEV distribution parameters with the same dimensions as `data`
        (excluding `core_dim`) and a new dimension `dparams`:
        If stationary, dparams = (c, loc, scale).
        If nonstationary, dparams = (c, loc0, loc1, scale0, scale1).

    Notes
    -----
    - Use `unpack_gev_params` to get the shape, location and scale parameters
    as a separate array. If nonstationary, the output will still be three
    parameters that have an extra covariate dimension.
    - For stationary data the parameters are estimated using
     `scipy.stats.genextreme.fit`.
    - For nonstationary data, the parameters (including the linear location and
    scale trend parameters are estimated by minimising
    a penalised negative log-likelihood function.
    - The `assert_good_fit` option ensures that the distribution fit is
    accepted if the goodness of fit test `p-value > alpha` (i.e., accept
    the null hypothesis). It will retry the fit using data[::2] to generate
    an initial guess.
    - The `covariate` must be numeric and have dimensions aligned with `data`.
    - If `pick_best_model` is a method, the relative goodness of fit method is
    used to determine if stationary or nonstationary parameters are returned.

    """
    kwargs = {k: v for k, v in locals().items() if k not in ["data", "covariate"]}

    def _assert_good_fit_1d(data, dparams, alpha, fit_kwargs):
        """Test goodness of stationary GEV fit and retry if failed."""
        pvalue = check_gev_fit(data, dparams)

        if np.all(pvalue < alpha):
            # Retry fit using alternative fitstart methods
            warnings.warn("GEV fit failed. Retrying fitstart with data subset.")
            _kwargs = fit_kwargs.copy()
            _kwargs["fitstart"] = _fitstart_1d(data[::2], fitstart)
            _kwargs["stationary"] = True
            dparams = _fit_1d(data, covariate, **_kwargs)
            pvalue = check_gev_fit(data, dparams)

        # Return NaNs if the test still fails
        if np.all(pvalue < alpha):
            # Return NaNs
            dparams = dparams * np.nan
            warnings.warn("Data fit failed.")
        return dparams

    def _fit_1d(
        data,
        covariate,
        stationary,
        fitstart,
        core_dim,
        loc1,
        scale1,
        assert_good_fit,
        pick_best_model,
        alpha,
        method,
    ):
        """Estimate distribution parameters."""
        if np.all(~np.isfinite(data)):
            # Return NaNs if all input data is infinite
            n = 3 if stationary else 5
            return np.array([np.nan] * n)

        if np.isnan(data).any():
            # Mask NaNs in data
            mask = np.isfinite(data)
            data = data[mask]
            if not stationary:
                covariate = covariate[mask]

        # Initial estimates of distribution parameters for MLE
        if isinstance(fitstart, str):
            dparams_i = _fitstart_1d(data, fitstart)
        else:
            # User provided initial estimates
            dparams_i = fitstart

        # Use genextreme to get stationary distribution parameters
        if stationary or pick_best_model:
            if dparams_i is None:
                dparams_i = genextreme.fit(data)
            else:
                dparams = genextreme.fit(
                    data, dparams_i[0], loc=dparams_i[1], scale=dparams_i[2]
                )
            dparams = np.array([i for i in dparams], dtype="float64")

            if assert_good_fit:
                dparams = _assert_good_fit_1d(data, dparams, alpha, kwargs)

        if not stationary or pick_best_model:
            # Temporarily reverse shape sign (scipy uses different sign convention)
            dparams_ns_i = [-dparams_i[0], dparams_i[1], loc1, dparams_i[2], scale1]

            # Optimisation bounds (scale parameter must be non-negative)
            bounds = [(None, None)] * 5
            bounds[3] = (0, None)  # Positive scale parameter
            if loc1 is None:
                dparams_ns_i[2] = 0
                bounds[2] = (0, 0)  # Only allow trend in scale
            if scale1 is None:
                dparams_ns_i[4] = 0
                bounds[4] = (0, 0)  # Only allow trend in location

            # Minimise the negative log-likelihood function to get optimal dparams
            res = minimize(
                nllf,
                dparams_ns_i,
                args=(data, covariate),
                method=method,
                bounds=bounds,
            )
            dparams_ns = np.array([i for i in res.x], dtype="float64")
            # Reverse shape sign for consistency with scipy.stats results
            dparams_ns[0] *= -1

            # Stationary and nonstationary model relative goodness of fit
            if pick_best_model:
                dparams = get_best_GEV_model_1d(
                    data, dparams, dparams_ns, covariate, alpha, test=pick_best_model
                )
            else:
                dparams = dparams_ns

        return dparams

    if covariate is not None:
        covariate = _format_covariate(data, covariate, core_dim)
    else:
        covariate = 0  # Dummy covariate for apply_ufunc

    # Input core dimensions
    if core_dim is not None and hasattr(covariate, core_dim):
        # Covariate has the same core dimension as data
        input_core_dims = [[core_dim], [core_dim]]
    else:
        # Covariate is a 1D array
        input_core_dims = [[core_dim], []]

    n_params = 5 if (not stationary or pick_best_model) else 3
    # Fit data to distribution parameters
    dparams = apply_ufunc(
        _fit_1d,
        data,
        covariate,
        input_core_dims=input_core_dims,
        output_core_dims=[["dparams"]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=["float64"],
        dask_gufunc_kwargs={"output_sizes": {"dparams": n_params}},
    )
    if isinstance(data, DataArray):
        # Format output (consistent with xclim)
        if n_params == 3:
            dparams.coords["dparams"] = ["c", "loc", "scale"]
        else:
            dparams.coords["dparams"] = ["c", "loc0", "loc1", "scale0", "scale1"]

        # Add coordinates for the distribution parameters
        dist_name = "genextreme" if stationary else "nonstationary genextreme"
        if isinstance(fitstart, str):
            estimator = fitstart.upper()
        else:
            estimator = f"User estimates = {fitstart}"

        dparams.attrs = dict(
            long_name=f"{dist_name.capitalize()} parameters",
            description=f"Parameters of the {dist_name} distribution",
            method="MLE",
            estimator=estimator,
            scipy_dist="genextreme",
            units="",
        )

    return dparams


def penalised_sum(x):
    """Sum finite values in a 1D array and add non-finite penalties.

    This is a utility function used when evaluating the negative
    log-likelihood for a distribution and an array of samples.
    Adapted/stolen from:
    https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_distn_infrastructure.py
    """
    finite_x = np.isfinite(x)
    bad_count = finite_x.size - np.count_nonzero(finite_x)

    total = np.sum(x[finite_x])
    penalty = bad_count * np.log(np.finfo(float).max) * 100

    return total + penalty


def nllf(dparams, x, covariate=None):
    """Penalised negative log-likelihood function.

    Parameters
    ----------
    dparams : tuple of floats
        Distribution parameters (stationary or non-stationary)
    x : array_like
        Data to use in estimating the distribution parameters
    covariate : array_like, optional
        Covariate used to estimate nonstationary parameters

    Returns
    -------
    total : float
        The penalised NLLF summed over all values of x

    Notes
    -----
    This is modified version of `scipy.stats.genextreme.fit` for fitting extreme
    value distribution parameters, in which the location and scale parameters
    can vary linearly with a covariate.
    The log-likelihood equations are based on Coles (2001; page 55).
    It is suitable for stationary or nonstationary distributions:
        - dparams = shape, loc, scale
        - dparams = shape, loc, loc1, scale, scale1
    The nonstationary parameters are returned if `dparams` incudes the location
    and scale trend parameters.
    A large finite penalty (instead of infinity) is applied for observations
    beyond the support of the distribution.
    The NLLF is not finite when the shape is nonzero and Z is negative because
    the PDF is zero (i.e., ``log(0)=inf)``).
    """
    if len(dparams) == 5:
        # Nonstationary GEV parameters
        shape, loc0, loc1, scale0, scale1 = dparams
        loc = loc0 + loc1 * covariate
        scale = scale0 + scale1 * covariate

    else:
        # Stationary GEV parameters
        shape, loc, scale = dparams

    s = (x - loc) / scale

    # Calculate the NLLF (type 1 or types 2-3 extreme value distributions)
    # Type I extreme value distributions (Gumbel)
    if np.fabs(shape) < 1e-6:
        valid = scale > 0
        L = np.log(scale, where=valid) + s + np.exp(-s)

    # Types II & III extreme value distributions (Fréchet and Weibull)
    else:
        Z = 1 + shape * s
        # The log-likelihood is finite when the shape and Z are positive
        valid = np.isfinite(Z) & (Z > 0) & (scale > 0)
        L = (
            np.log(scale, where=valid)
            + (1 + 1 / shape) * np.log(Z, where=valid)
            + np.power(Z, -1 / shape, where=valid)
        )

    L = np.where(valid, L, np.inf)

    # Sum function (where finite) & add penalty for each infinite element
    total = penalised_sum(L)
    return total


def _fitstart_1d(data, method):
    """Generate initial parameter guesses for nonstationary fit.

    Parameters
    ----------
    data : array_like
        Data to use in estimating the distribution parameters
    method : {'LMM', 'scipy_fitstart', 'scipy', 'scipy_subset',
    'xclim_fitstart', 'xclim'}
        Initial guess method of the shape, loc and scale parameters

    Returns
    -------
    dparams_i : list
        Initial guess of the shape, loc and scale parameters

    Notes
    -----
    - Use `scipy_fitstart` to reproduce the scipy fit in `fit_gev`.
    - The LMM shape sign is reversed for consistency with scipy.stats results.
    """

    if method == "LMM":
        # L-moments method
        dparams_i = distr.gev.lmom_fit(data)
        dparams_i = list(dparams_i.values())
        dparams_i[0] = -dparams_i[0]

    elif method == "scipy_fitstart":
        # Moments method?
        dparams_i = genextreme._fitstart(data)

    elif method == "scipy":
        # MLE
        dparams_i = genextreme.fit(data)

    elif method == "scipy_subset":
        # MLE (equivalent of fitstart='scipy_subet')
        dparams_i = genextreme.fit(data[::2])

    elif method == "xclim_fitstart":
        # Approximates the probability weighted moments (PWM) method?
        args, kwargs = xcstats._fit_start(data, dist="genextreme")
        dparams_i = [args[0], kwargs["loc"], kwargs["scale"]]

    elif method == "xclim":
        # MLE
        da = DataArray(data, dims="time")
        dparams_i = xcstats.fit(da, "genextreme", method="MLE")
    else:
        raise ValueError(f"Unknown fitstart method: {method}")

    return np.array(dparams_i, dtype="float64")


def _format_covariate(data, covariate, core_dim):
    """Format or generate covariate.

    Parameters
    ----------
    data : xarray.DataArray
        Data to use in estimating the distribution parameters
    covariate : array_like or str
        A nonstationary covariate array or coordinate name
    core_dim : str
        Name of time/sample dimension in `data`

    Returns
    -------
    covariate : xarray.DataArray
        Covariate with the same core_dim as data
    """

    if isinstance(covariate, str):
        # Select coordinate in data
        covariate = data[covariate]

    elif covariate is None:
        # Guess covariate
        if core_dim in data:
            covariate = data[core_dim]
        else:
            covariate = np.arange(data.shape[0])

    if covariate.dtype.kind not in set("buifc"):
        # Convert dates to numbers
        covariate = date2num(covariate)

    if not isinstance(covariate, DataArray):
        # Convert to DataArray with the same core_dim as data
        covariate = DataArray(covariate, dims=[core_dim])

    return covariate


def check_gev_fit(data, dparams, core_dim=[], **kwargs):
    """Test stationary GEV distribution goodness of fit.

    Parameters
    ----------
    data: array_like
        Data used to estimate the distribution parameters
    dparams : tuple of floats
        Shape, location and scale parameters
    core_dim : str, optional
        Data dimension to test over
    kwargs : dict, optional
        Additional keyword arguments to pass to `goodness_of_fit`.

    Returns
    -------
    pvalue : scipy.stats._fit.GoodnessOfFitResult.pvalue
        Goodness of fit p-value
    """

    def _goodness_of_fit(data, dparams, **kwargs):
        """Test GEV goodness of fit."""
        # Stationary parameters
        shape, loc, scale = dparams

        res = goodness_of_fit(
            genextreme,
            data,
            known_params=dict(c=shape, loc=loc, scale=scale),
            **kwargs,
        )
        return res.pvalue

    if not isinstance(core_dim, list):
        core_dim = [core_dim]

    pvalue = apply_ufunc(
        _goodness_of_fit,
        data,
        dparams,
        input_core_dims=[core_dim, ["dparams"]],
        vectorize=True,
        kwargs=kwargs,
        dask="parallelized",
        dask_gufunc_kwargs=dict(meta=(np.ndarray(1, float),)),
    )
    return pvalue


def check_gev_relative_fit(data, L1, L2, test, alpha=0.05):
    """Test relative fit of stationary and nonstationary GEV distribution.

    Parameters
    ----------
    data : array_like
        Data to use in estimating the distribution parameters
    L1, L2 : float
        Negative log-likelihood of the stationary and nonstationary model
    test : {"AIC", "BIC", "LRT"}
        Method to test relative fit of stationary and nonstationary models

    Returns
    -------
    result : bool
        If True, the nonstationary model is better

    Notes
    -----
    For more information on the tests see:
    Kim, H., Kim, S., Shin, H., & Heo, J. (2017). Appropriate model selection
    methods for nonstationary generalized extreme value models. Journal of
    Hydrology, 547, 557-574. https://doi.org/10.1016/j.jhydrol.2017.02.005
    """

    dof = [3, 5]  # Degrees of freedom of each model

    if test.casefold() == "lrt":
        # Likelihood ratio test statistic
        LR = -2 * (L2 - L1)
        result = chi2.sf(LR, dof[1] - dof[0]) <= alpha

    elif test.casefold() == "aic":
        # Akaike Information Criterion (AIC)
        aic = [(2 * k) + (2 * n) for n, k in zip([L1, L2], dof)]
        result = aic[0] > aic[1]

    elif test.casefold() == "bic":
        # Bayesian Information Criterion (BIC)
        bic = [k * np.log(len(data)) + (2 * n) for n, k in zip([L1, L2], dof)]
        result = bic[0] > bic[1]
    else:
        raise ValueError("test must be 'LRT', 'AIC' or 'BIC'", test)
    return result


def get_best_GEV_model_1d(data, dparams, dparams_ns, covariate, alpha, test):
    """Get the best GEV model based on a relative fit test."""
    # Calculate the stationary GEV parameters
    shape, loc, scale = dparams

    # Negative log-likelihood of stationary and nonstationary models
    L1 = nllf([-shape, loc, scale], data)
    L2 = nllf([-dparams_ns[0], *dparams_ns[1:]], data, covariate)

    result = check_gev_relative_fit(data, L1, L2, test=test, alpha=alpha)
    if not result:
        # Return the stationary parameters with no trend
        dparams = np.array([shape, loc, 0, scale, 0], dtype="float64")
    else:
        dparams = dparams_ns
    return dparams


def unpack_gev_params(dparams, covariate=None):
    """Unpack shape, loc, scale from dparams.

    Parameters
    ----------
    dparams : xarray.DataArray, list or tuple
        Stationary or nonstationary GEV parameters
    covariate : xarray.DataArray, optional
        Covariate values for nonstationary parameters

    Returns
    -------
    shape, loc, scale : array_like or float
        GEV parameters. If nonstationary, loc and scale are functions of the
        covariate.
    """

    if hasattr(dparams, "dparams"):
        # Select the correct dimension in a DataArray
        dparams = [dparams.isel(dparams=i) for i in range(dparams.dparams.size)]
    elif not isinstance(dparams, (list, tuple)) and dparams.ndim > 1:
        warnings.warn(f"Assuming parameters on axis=-1 (shape={dparams.shape})")
        dparams = np.split(dparams, dparams.shape[-1], axis=-1)

    # Unpack GEV parameters
    if len(dparams) == 3:
        # Stationary GEV parameters
        shape, loc, scale = dparams

    elif len(dparams) == 5:
        # Nonstationary GEV parameters
        shape, loc0, loc1, scale0, scale1 = dparams
        loc = loc0 + loc1 * covariate
        scale = scale0 + scale1 * covariate
    else:
        raise ValueError("Expected 3 or 5 GEV parameters.", dparams)

    return shape, loc, scale


def get_return_period(event, dparams=None, covariate=None, **kwargs):
    """Get return periods for a given events.

    Parameters
    ----------
    event : float or array_like
        Event value(s) for which to calculate the return period
    dparams : array_like, optional
        Stationary or nonstationary GEV parameters
    covariate : array_like, optional
        Covariate values for nonstationary parameters
    kwargs : dict, optional
        Additional keyword arguments to pass to `fit_gev`

    Returns
    -------
    return_period : float or array_like
        Return period(s) for the event(s)
    """

    if dparams is None:
        dparams = fit_gev(**kwargs)

    shape, loc, scale = unpack_gev_params(dparams, covariate)

    probability = apply_ufunc(
        genextreme.sf,
        event,
        shape,
        loc,
        scale,
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
    )
    return 1.0 / probability


def get_return_level(return_period, dparams=None, covariate=None, **kwargs):
    """Get the return levels for given return periods.

    Parameters
    ----------
    return_period : float or array_like
       Return period(s) for which to calculate the return level
    dparams : array_like, optional
        Stationary or nonstationary GEV parameters
    covariate : array_like, optional
        Covariate values for nonstationary parameters
    kwargs : dict, optional
        Additional keyword arguments to pass to `fit_gev`

    Returns
    -------
    return_level : float or array_like
        Return level(s) of the given return period(s)

    Notes
    -----
    If `return_period` is an ndarray, make sure dimensions are aligned with
    `dparams`. For example, dparams dims=(lat, lon, dparams) and return_period
    dims=(lat, lon, period).
    """

    if dparams is None:
        dparams = fit_gev(**kwargs)

    shape, loc, scale = unpack_gev_params(dparams, covariate)

    return_level = apply_ufunc(
        genextreme.isf,
        1 / return_period,
        shape,
        loc,
        scale,
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
    )
    return return_level


def gev_return_curve(
    data,
    event_value,
    bootstrap_method="non-parametric",
    n_bootstraps=1000,
    max_return_period=4,
    max_shape_ratio=None,
    **fit_kwargs,
):
    """Return x and y data for a GEV return period curve.

    Parameters
    ----------
    data : xarray.DataArray
    event_value : float
        Magnitude of event of interest
    bootstrap_method : {'parametric', 'non-parametric'}, default "non-parametric"
    n_bootstraps : int, default 1000
    max_return_period : float, default 4
        The maximum return period is 10^{max_return_period}
    max_shape_ratio: float, optional
        Maximum bootstrap shape parameter to full population shape parameter
        ratio (e.g. 6.0). Useful for filtering bad fits to bootstrap samples
    fit_kwargs : dict, optional
        Additional keyword arguments to pass to `fit_gev`
    """
    rng = np.random.default_rng(seed=0)

    # GEV fit to data
    dparams = fit_gev(data, **fit_kwargs)
    shape, loc, scale = unpack_gev_params(dparams)

    curve_return_periods = np.logspace(0, max_return_period, num=10000)
    curve_probabilities = 1.0 / curve_return_periods
    curve_values = genextreme.isf(curve_probabilities, shape, loc, scale)

    event_probability = genextreme.sf(event_value, shape, loc=loc, scale=scale)
    event_return_period = 1.0 / event_probability

    # Bootstrapping for confidence interval
    boot_values = curve_values
    boot_event_return_periods = []
    for i in range(n_bootstraps):
        if bootstrap_method == "parametric":
            boot_data = genextreme.rvs(shape, loc=loc, scale=scale, size=len(data))
        elif bootstrap_method == "non-parametric":
            boot_data = rng.choice(data, size=data.shape, replace=True)
        boot_shape, boot_loc, boot_scale = fit_gev(boot_data, fitstart="scipy_subet")
        if max_shape_ratio:
            shape_ratio = abs(boot_shape) / abs(shape)
            if shape_ratio > max_shape_ratio:
                continue
        boot_value = genextreme.isf(
            curve_probabilities, boot_shape, boot_loc, boot_scale
        )
        boot_values = np.vstack((boot_values, boot_value))

        boot_event_probability = genextreme.sf(
            event_value, boot_shape, loc=boot_loc, scale=boot_scale
        )
        boot_event_return_period = 1.0 / boot_event_probability
        boot_event_return_periods.append(boot_event_return_period)

    curve_values_lower_ci = np.quantile(boot_values, 0.025, axis=0)
    curve_values_upper_ci = np.quantile(boot_values, 0.975, axis=0)
    curve_data = (
        curve_return_periods,
        curve_values,
        curve_values_lower_ci,
        curve_values_upper_ci,
    )

    boot_event_return_periods = np.array(boot_event_return_periods)
    boot_event_return_periods = boot_event_return_periods[
        np.isfinite(boot_event_return_periods)
    ]
    event_return_period_lower_ci = np.quantile(boot_event_return_periods, 0.025)
    event_return_period_upper_ci = np.quantile(boot_event_return_periods, 0.975)
    event_data = (
        event_return_period,
        event_return_period_lower_ci,
        event_return_period_upper_ci,
    )

    return curve_data, event_data


def plot_gev_return_curve(
    ax,
    data,
    event_value,
    direction="exceedance",
    bootstrap_method="parametric",
    n_bootstraps=1000,
    max_return_period=4,
    ylabel=None,
    ylim=None,
    text=False,
    max_shape_ratio=None,
    **fit_kwargs,
):
    """Plot a single return period curve.

    Parameters
    ----------
    ax : matplotlib plot axis
    data : xarray.DataArray
    event_value : float
        Magnitude of the event of interest
    direction : {'exceedance', 'deceedance'}, default 'exceedance'
        Plot exceedance or deceedance probabilities
    bootstrap_method : {'parametric', 'non-parametric'}, default 'non-parametric'
    n_bootstraps : int, default 1000
    max_return_period : float, default 4
        The maximum return period is 10^{max_return_period}
    ylabel : str, optional
        Text for y axis label
    ylim : float, optional
        Limits for y-axis
    text : bool, default False
       Write the return period (and 95% CI) on the plot
    max_shape_ratio: float, optional
        Maximum bootstrap shape parameter to full population shape parameter ratio (e.g. 6.0)
        Useful for filtering bad fits to bootstrap samples
    fit_kwargs : dict, optional
        Additional keyword arguments to pass to `fit_gev`
    """

    if direction == "deceedance":
        ValueError("Deceedance functionality not implemented yet")

    curve_data, event_data = gev_return_curve(
        data,
        event_value,
        bootstrap_method=bootstrap_method,
        n_bootstraps=n_bootstraps,
        max_return_period=max_return_period,
        max_shape_ratio=max_shape_ratio,
        **fit_kwargs,
    )
    (
        curve_return_periods,
        curve_values,
        curve_values_lower_ci,
        curve_values_upper_ci,
    ) = curve_data
    (
        event_return_period,
        event_return_period_lower_ci,
        event_return_period_upper_ci,
    ) = event_data

    ax.plot(
        curve_return_periods,
        curve_values,
        color="tab:blue",
        label="GEV fit to data",
    )
    ax.fill_between(
        curve_return_periods,
        curve_values_lower_ci,
        curve_values_upper_ci,
        color="tab:blue",
        alpha=0.2,
        label="95% CI on GEV fit",
    )
    ax.plot(
        [event_return_period_lower_ci, event_return_period_upper_ci],
        [event_value] * 2,
        color="0.5",
        marker="|",
        linestyle=":",
        label="95% CI for record event",
    )
    empirical_return_values = np.sort(data, axis=None)[::-1]
    empirical_return_periods = len(data) / np.arange(1.0, len(data) + 1.0)
    ax.scatter(
        empirical_return_periods,
        empirical_return_values,
        color="tab:blue",
        alpha=0.5,
        label="empirical data",
    )
    rp = f"{event_return_period:.0f}"
    rp_lower = f"{event_return_period_lower_ci:.0f}"
    rp_upper = f"{event_return_period_upper_ci:.0f}"
    if text:
        ax.text(
            0.98,
            0.05,
            f"{rp} ({rp_lower}-{rp_upper}) years",
            transform=ax.transAxes,
            color="black",
            horizontalalignment="right",
            fontsize="large",
        )
    else:
        print(f"{rp} year return period")
        print(f"95% CI: {rp_lower}-{rp_upper} years")

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[3], handles[0], handles[1], handles[2]]
    labels = [labels[3], labels[0], labels[1], labels[2]]
    ax.legend(handles, labels, loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("return period (years)")
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid()


def plot_nonstationary_pdfs(
    data,
    dparams_s,
    dparams_ns,
    covariate,
    ax=None,
    title="",
    units=None,
    cmap="rainbow",
    outfile=None,
):
    """Plot stationary and nonstationary GEV PDFs.

    Parameters
    ----------
    data : array-like
        Data to plot the histogram
    dparams_s : tuple of floats
        Stationary GEV parameters (shape, loc, scale)
    dparams_ns : tuple or array-like
        Nonstationary GEV parameters (shape, loc0, loc1, scale0, scale1)
    covariate : array-like
        Covariate values in which to plot the nonstationary PDFs
    ax : matplotlib.axes.Axes
    title : str, optional
    xlabel : str, optional
    cmap : str, default "rainbow"
    outfile : str, optional

    Returns
    -------
    ax : matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.set_title(title, loc="left")

    n = covariate.size
    colors = colormaps[cmap](np.linspace(0, 1, n))
    shape, loc, scale = unpack_gev_params(dparams_ns, covariate)

    # Histogram.
    _, bins, _ = ax.hist(data, bins=40, density=True, alpha=0.5, label="Histogram")

    # Stationary GEV PDF
    shape_s, loc_s, scale_s = dparams_s
    pdf_s = genextreme.pdf(bins, shape_s, loc=loc_s, scale=scale_s)
    ax.plot(bins, pdf_s, c="k", ls="--", lw=2.8, label="Stationary")

    # Nonstationary GEV PDFs
    for i, t in enumerate(covariate.values):
        pdf_ns = genextreme.pdf(bins, shape, loc=loc[i], scale=scale[i])
        ax.plot(bins, pdf_ns, lw=1.6, c=colors[i], zorder=0, label=t)

    ax.set_xlabel(units)
    ax.set_ylabel("Probability")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), framealpha=0.3)
    ax.set_xmargin(1e-3)

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    return ax


def plot_nonstationary_return_curve(
    return_periods,
    dparams_s,
    dparams_ns,
    covariate,
    dim="time",
    ax=None,
    title="",
    units=None,
    cmap="rainbow",
    outfile=None,
):
    """Plot stationary and nonstationary return period curves.

    Parameters
    ----------
    return_periods : array-like
        Return periods to plot (x-axis)
    dparams_s : array-like or tuple of floats
        Stationary GEV parameters (shape, loc, scale)
    dparams_ns : array-like or tuple of floats
        Nonstationary GEV parameters (shape, loc0, loc1, scale0, scale1)
    covariate : array-like
        Covariate values in which to show the nonstationary return levels
    dim : str, optional
        Covariate core dimension name, default "time"
    ax : matplotlib.axes.Axes
    title : str, optional
    units : str, optional
    cmap : str, default "rainbow"
    outfile : str, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    ax.set_title(title, loc="left")

    n = covariate.size
    colors = colormaps[cmap](np.linspace(0, 1, n))

    # Stationary return levels
    if dparams_s is not None:
        return_levels = get_return_level(return_periods, dparams_s)
        ax.plot(
            return_periods,
            return_levels,
            label="Stationary",
            c="k",
            ls="--",
            zorder=n + 1,
        )

    # Nonstationary return levels
    return_levels = get_return_level(return_periods, dparams_ns, covariate)
    for i, t in enumerate(covariate.values):
        ax.plot(return_periods, return_levels.isel({dim: i}), label=t, c=colors[i])

    ax.set_xscale("log")
    ax.set_ylabel(units)
    ax.set_xlabel("Return period")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xmargin(1e-2)
    ax.legend()

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    return ax


def plot_stacked_histogram(
    dv1,
    dv2,
    bins=None,
    labels=None,
    ax=None,
    title="",
    units=None,
    cmap="rainbow",
    legend=True,
    outfile=None,
):
    """Histogram with data binned and stacked.

    Parameters
    ----------

    dv1 : xarray.DataArray
        Data to plot in the histogram
    dv2 : xarray.DataArray
        Covariate used to bin the data
    bins : array-like
        Bin edges of dv2
    labels : array-like, optional
        Labels for each bin, default None uses left side of each bin
    dim : str, default "time"
        Core dimension name of dv1 and dv2
    ax : matplotlib.axes.Axes
    title : str, optional
    units : str, optional
    cmap : str, default "rainbow"
    legend : bool, optional
    outfile : str, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    assert dv1.size == dv2.size

    if bins is None or np.ndim(bins) == 0:
        bins = np.histogram_bin_edges(dv2, bins)

        # Round bins to integers if possible
        if np.all(np.diff(bins) >= 1):
            bins = np.ceil(bins).astype(dtype=int)

    if labels is None:
        # Labels show left side of each bin
        # labels = bins[:-1]
        labels = [f"{bins[i]}-{bins[i+1] - 1}" for i in range(len(bins) - 1)]

    # Subset dv1 by bins
    dx_subsets = [
        dv1.where(((dv2 >= bins[a]) & (dv2 < bins[a + 1])).values)
        for a in range(len(bins) - 1)
    ]
    dx_subsets[-1] = dv1.where((dv2 >= bins[-2]).values)

    colors = colormaps[cmap](np.linspace(0, 1, len(bins) - 1))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.set_title(title, loc="left")
    ax.hist(
        dx_subsets,
        density=True,
        stacked=True,
        histtype="barstacked",
        color=colors,
        edgecolor="k",
        label=labels,
    )
    if legend:
        ax.legend()
    ax.set_xlabel(units)
    ax.set_ylabel("Probability")

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    return ax, bins


def _parse_command_line():
    """Parse the command line for input arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("file", type=str, help="Forecast file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument(
        "--stack_dims",
        type=str,
        nargs="*",
        default=["ensemble", "init_date", "lead_time"],
        help="Dimensions to stack",
    )
    parser.add_argument("--core_dim", type=str, default="time", help="Core dimension")
    parser.add_argument(
        "--stationary",
        type=bool,
        default=True,
        help="Fit nonstationary GEV distribution",
    )
    parser.add_argument(
        "--fitstart",
        default="LMM",
        choices=(
            "LMM",
            "scipy",
            "scipy_fitstart",
            "scipy_subset",
            "xclim_MLE",
            "xclim",
            ["shape", "loc", "scale"],
        ),
        help="Initial guess method (or estimate) of the GEV parameters",
    )
    parser.add_argument(
        "--assert_good_fit",
        action="store_true",
        default=False,
        help="Test fit goodness",
    )
    parser.add_argument(
        "--pick_best_model",
        type=str,
        default=None,
        help="Relative fit test to pick stationary or nonstationary parameters",
    )
    parser.add_argument(
        "--reference_time_period",
        type=str,
        nargs=2,
        default=None,
        help="Reference time period (start_date, end_date)",
    )
    parser.add_argument(
        "--covariate", type=str, default="time.year", help="Covariate variable"
    )
    parser.add_argument(
        "--covariate_file", type=str, default=None, help="Covariate file"
    )
    parser.add_argument(
        "--min_lead", default=None, help="Minimum lead time (int or filename)"
    )
    parser.add_argument(
        "--min_lead_kwargs",
        type=str,
        nargs="*",
        default={},
        action=general_utils.store_dict,
        help="Minimum lead time file",
    )
    # parser.add_argument(
    #     "--confidence_interval",
    #     type=float,
    #     default=0.95,
    #     help="Confidence interval e.g., --confidence_interval 0.95",
    # )
    parser.add_argument(
        "--ensemble_dim",
        type=str,
        default="ensemble",
        help="Name of ensemble member dimension",
    )
    parser.add_argument(
        "--init_dim",
        type=str,
        default="init_date",
        help="Name of initial date dimension",
    )
    parser.add_argument(
        "--lead_dim",
        type=str,
        default="lead_time",
        help="Name of lead time dimension",
    )
    parser.add_argument(
        "--output_chunks",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        default={},
        help="Output chunks",
    )
    parser.add_argument(
        "--dask_config", type=str, help="YAML file specifying dask client configuration"
    )
    args = parser.parse_args()

    return args


def _main():
    """Run the command line program."""

    args = _parse_command_line()

    ds = fileio.open_dataset(args.file, variables=[args.var])

    if args.covariate_file is not None:
        # Add covariate to dataset (to ensure all operations are aligned)
        ds_covariate = fileio.open_dataset(
            args.covariate_file, variables=[args.covariate]
        )
        ds[args.covariate] = ds_covariate[args.covariate]

    # Filter data by reference time period
    if args.reference_time_period:
        ds = time_utils.select_time_period(ds, args.reference_time_period)

    # Filter data by minimum lead time
    if args.min_lead:
        if isinstance(args.min_lead, str):
            # Load min_lead from file
            ds_min_lead = fileio.open_dataset(args.min_lead, **args.min_lead_kwargs)
            min_lead = ds_min_lead["min_lead"].load()
            ds = ds.groupby(f"{args.init_dim}.month").where(
                ds[args.lead_dim] >= min_lead
            )
            ds = ds.drop_vars("month")
        else:
            ds = ds.where(ds[args.lead_dim] >= args.min_lead)

    # Stack dimensions along new "sample" dimension
    if all([dim in ds[args.var].dims for dim in args.stack_dims]):
        ds = ds.stack(**{"sample": args.stack_dims})
        args.core_dim = "sample"

    if not args.stationary:
        covariate = _format_covariate(ds[args.var], ds[args.covariate], args.core_dim)
    else:
        covariate = None

    dparams = fit_gev(
        ds[args.var],
        core_dim=args.core_dim,
        stationary=args.stationary,
        fitstart=args.fitstart,
        covariate=covariate,
        assert_good_fit=args.assert_good_fit,
        pick_best_model=args.pick_best_model,
    )

    # Format outfile
    dparams = dparams.to_dataset()

    # Add the covariate variable
    if not args.stationary or args.pick_best_model:
        dparams[args.covariate] = covariate

    infile_logs = {args.file: ds.attrs["history"]}
    if isinstance(args.min_lead, str):
        infile_logs[args.min_lead] = ds_min_lead.attrs["history"]
    dparams.attrs["history"] = fileio.get_new_log(infile_logs=infile_logs)

    if args.output_chunks:
        dparams = dparams.chunk(args.output_chunks)

    if "zarr" in args.outfile:
        fileio.to_zarr(dparams, args.outfile)
    else:
        dparams.to_netcdf(args.outfile, compute=True)


if __name__ == "__main__":
    _main()
