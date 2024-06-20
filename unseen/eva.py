"""Extreme value analysis functions."""

from matplotlib.dates import date2num
import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme, goodness_of_fit
from scipy.stats.distributions import chi2
import warnings
from xarray import apply_ufunc, DataArray


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


def fit_stationary_gev(x, user_estimates=[], generate_estimates=False):
    """Estimate stationary shape, location and scale parameters.

    Parameters
    ----------
    x : array_like
         Data to use in estimating the distribution parameters
    user_estimates : list, optional
        Initial guess of the shape, loc and scale parameters
    generate_estimates : bool, optional
        Generate initial parameter guesses using a data subset

    Returns
    -------
    shape, loc, scale : float
        GEV parameters
    """

    if any(user_estimates):
        shape, loc, scale = user_estimates
        shape, loc, scale = genextreme.fit(x, shape, loc=loc, scale=scale)

    elif generate_estimates:
        # Generate initial estimates using a data subset (useful for large datasets)
        shape, loc, scale = genextreme.fit(x[::2])
        shape, loc, scale = genextreme.fit(x, shape, loc=loc, scale=scale)
    else:
        shape, loc, scale = genextreme.fit(x)
    return shape, loc, scale


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


def nllf(theta, x, covariate=None):
    """Penalised negative log-likelihood function.

    Parameters
    ----------
    theta : tuple of floats
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
    The log-likelihood equations are based on Méndez et al. (2007).
    It is suitable for stationary or nonstationary distributions:
        - theta = shape, loc, scale
        - theta = shape, loc, loc1, scale, scale1
    The nonstationary parameters are returned if `theta` incudes the location
    and scale trend parameters.
    A large finite penalty (instead of infinity) is applied for observations
    beyond the support of the distribution.
    The NLLF is not finite when the shape is nonzero and Z is negative because
    the PDF is zero (i.e., ``log(0)=inf)``).
    """
    if len(theta) == 5:
        # Nonstationary GEV parameters
        shape, loc0, loc1, scale0, scale1 = theta
        loc = loc0 + loc1 * covariate
        scale = scale0 + scale1 * covariate

    else:
        # Stationary GEV parameters
        shape, loc, scale = theta

    s = (x - loc) / scale

    # Calculate the NLLF (type 1 or types 2-3 extreme value distributions)
    # Type I extreme value distributions (Gumbel)
    if shape == 0:
        valid = scale > 0
        L = np.log(scale, where=valid) + s + np.exp(-s)

    # Types II & III extreme value distributions (Fréchet and Weibull)
    else:
        Z = 1 + shape * s
        # The log-likelihood function is finite when the shape and Z are positive
        valid = np.isfinite(Z) & (Z > 0) & (scale > 0)
        L = (
            np.log(scale, where=valid)
            + (1 + 1 / shape) * np.log(Z, where=valid)
            + np.power(Z, -1 / shape, where=valid)
        )

    L = np.where(valid, L, np.inf)

    # Sum function along all axes (where finite) & add penalty for each infinite element
    total = penalised_sum(L)
    return total


def fit_gev(
    data,
    core_dim="time",
    stationary=True,
    covariate=None,
    loc1=0,
    scale1=0,
    test_fit_goodness=False,
    relative_fit_test=None,
    alpha=0.05,
    user_estimates=[],
    generate_estimates=False,
    method="Nelder-Mead",
):
    """Estimate stationary or nonstationary GEV distribution parameters.

    Parameters
    ----------
    data : array_like
        Data to use in estimating the distribution parameters
    core_dim : str, optional
        Name of time/sample dimension in `data`. Default: "time".
    stationary : bool, optional
        Fit as a stationary GEV using `fit_stationary_gev`. Default: True.
    covariate : array_like or str, optional
        A nonstationary covariate array or coordinate name
    loc1, scale1 : float or None, optional
        Initial guess of trend parameters. If None, the trend is fixed at zero.
    test_fit_goodness : bool, optional
        Test goodness of fit and attempt retry. Default False.
    relative_fit_test : {None, 'lrt', 'aic', 'bic'}, optional
        Method to test relative fit of stationary and nonstationary models.
        The trend parameters are set to zero if the stationary fit is better.
    alpha : float, optional
        Goodness of fit p-value threshold. Default 0.05.
    user estimates: list, optional
        Initial guess of the shape, loc and scale parameters
    generate_estimates : bool, optional
        Generate initial parameter guesses using a data subset
    method : str, optional
        Optimization method for nonstationary fit {'Nelder-Mead', 'L-BFGS-B',
        'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA'}.

    Returns
    -------
    theta : xr.DataArray
        The GEV distribution parameters with the same dimensions as `data`
        (excluding `core_dim`) and a new dimension `theta`:
        If stationary, theta = (shape, loc, scale).
        If nonstationary, theta = (shape, loc0, loc1, scale0, scale1).

    Notes
    -----
    For stationary data the shape, location and scale parameters are
    estimated using `gev_stationary_fit`.
    For nonstationary data, the linear location and scale trend
    parameters are estimated using a penalized negative log-likelihood
    function with initial estimates based on the stationary fit.
    The distribution fit is considered good if the p-value is above
     `alpha` (i.e., accept the null hypothesis). Otherwise, it retry the fit
    without `user_estimates` and with `generating_estimates`.
    If data is a stacked forecast ensemble, the covariate may need to be
    stacked in the same way.
    """
    kwargs = locals()  # Function inputs

    def _format_covariate(data, covariate, stationary, core_dim):
        """Format or generate covariate ."""
        if not stationary:
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
        else:
            covariate = 0  # Dummy covariate for apply_ufunc

        return covariate

    def _fit(
        data,
        covariate,
        core_dim,
        user_estimates,
        generate_estimates,
        loc1,
        scale1,
        stationary,
        test_fit_goodness,
        relative_fit_test,
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

        # Use genextreme to get stationary distribution parameters
        theta = fit_stationary_gev(data, user_estimates, generate_estimates)

        if not stationary:
            # Use genextreme as initial guesses
            shape, loc, scale = theta
            # Temporarily reverse shape sign (scipy uses different sign convention)
            theta_i = [-shape, loc, loc1, scale, scale1]

            # Optimisation bounds (scale parameter must be non-negative)
            bounds = [(None, None)] * len(theta_i)
            bounds[3] = (0, None)  # Positive scale parameter
            if loc1 is None:
                theta_i[2] = 0
                bounds[2] = (0, 0)  # Only allow trend in scale
            if scale1 is None:
                theta_i[4] = 0
                bounds[4] = (0, 0)  # Only allow trend in location

            # Minimise the negative log-likelihood function to get optimal theta
            res = minimize(
                nllf,
                theta_i,
                args=(data, covariate),
                method=method,
                bounds=bounds,
            )
            theta = res.x

            if isinstance(relative_fit_test, str):
                # Test relative fit of stationary and nonstationary models
                # Negative log likelihood using genextreme parameters
                L1 = nllf([-shape, loc, scale], data)
                L2 = res.fun

                result = check_gev_relative_fit(
                    data, L1, L2, test=relative_fit_test, alpha=alpha
                )
                if result is False:
                    warnings.warn(
                        f"{relative_fit_test} test failed. Returning stationary parameters."
                    )
                    # Return stationary parameters (genextreme.fit output) with no trend
                    theta = [shape, loc, 0, scale, 0]

            # Reverse shape sign for consistency with scipy.stats results
            theta[0] *= -1

        theta = np.array([i for i in theta], dtype="float64")

        if test_fit_goodness and stationary:
            pvalue = check_gev_fit(data, theta, core_dim=core_dim)

            # Accept null distribution of the Anderson-darling test (same distribution)
            if np.all(pvalue < alpha):
                if any(kwargs["user_estimates"]):
                    warnings.warn("GEV fit failed. Retrying without user_estimates.")
                    kwargs["user_estimates"] = [None, None, None]
                    theta = _fit(data, covariate, **kwargs)
                elif not kwargs["generate_estimates"]:
                    warnings.warn(
                        "GEV fit failed. Retrying with generate_estimates=True."
                    )
                    kwargs["generate_estimates"] = True  # Also breaks loop
                    theta = _fit(data, covariate, **kwargs)
                else:
                    # Return NaNs
                    theta = theta * np.nan
                    warnings.warn("Data fit failed.")
        return theta

    data = kwargs.pop("data")
    covariate = kwargs.pop("covariate")
    covariate = _format_covariate(data, covariate, stationary, core_dim)

    # Input core dimensions
    if hasattr(covariate, core_dim):
        # Covariate has the same core dimension as data
        input_core_dims = [[core_dim], [core_dim]]
    else:
        # Covariate is a 1D array
        input_core_dims = [[core_dim], []]

    # Expected output of theta
    n = 3 if stationary else 5

    # Fit data to distribution parameters
    theta = apply_ufunc(
        _fit,
        data,
        covariate,
        input_core_dims=input_core_dims,
        output_core_dims=[["theta"]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=["float64"],
        dask_gufunc_kwargs=dict(output_sizes={"theta": n}),
    )

    # Format output
    if len(data.shape) == 1:
        # Return a tuple of scalars instead of a data array
        theta = np.array([i for i in theta], dtype="float64")

    if isinstance(theta, DataArray):
        if stationary:
            coords = ["shape", "loc", "scale"]
        else:
            coords = ["shape", "loc0", "loc1", "scale0", "scale1"]
        theta.coords["theta"] = coords
    return theta


def check_gev_fit(data, params, core_dim="time", **kwargs):
    """Test stationary GEV distribution goodness of fit.

    Parameters
    ----------
    data: array_like
        Data used to estimate the distribution parameters
    params : tuple of floats
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

    def _goodness_of_fit(data, params, **kwargs):
        """Test GEV goodness of fit."""
        # Stationary parameters
        shape, loc, scale = params

        res = goodness_of_fit(
            genextreme,
            data,
            known_params=dict(c=shape, loc=loc, scale=scale),
            **kwargs,
        )
        return res.pvalue

    pvalue = apply_ufunc(
        _goodness_of_fit,
        data,
        params,
        input_core_dims=[[core_dim], ["theta"]],
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
    test : {"aic", "bic", "lrt"}
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
    return result


def return_period(data, event, params=None, **kwargs):
    """Get return period for given event by fitting a GEV."""

    # GEV fit to data
    if len(params) != 3 or not kwargs.get("stationary", True):
        raise NotImplementedError(
            "Non-stationary GEV parameters must be evaluated at a point first."
        )

    if params is None:
        params = fit_gev(data, **kwargs)
    shape, loc, scale = params
    return_period = genextreme.isf(event, shape, loc=loc, scale=scale)

    return return_period


def gev_return_curve(
    data,
    event_value,
    bootstrap_method="non-parametric",
    n_bootstraps=1000,
    max_return_period=4,
):
    """Return x and y data for a GEV return period curve.

    Parameters
    ----------
    data : xarray DataArray
    event_value : float
        Magnitude of event of interest
    bootstrap_method : {'parametric', 'non-parametric'}, default 'non-parametric'
    n_bootstraps : int, default 1000
    max_return_period : float, default 4
        The maximum return period is 10^{max_return_period}
    """

    # GEV fit to data
    shape, loc, scale = fit_gev(data, generate_estimates=True, stationary=True)

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
            boot_data = np.random.choice(data, size=data.shape, replace=True)
        boot_shape, boot_loc, boot_scale = fit_gev(boot_data, generate_estimates=True)

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
):
    """Plot a single return period curve.

    Parameters
    ----------
    ax : matplotlib plot axis
    data : xarray DataArray
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
    """

    if direction == "deceedance":
        ValueError("Deceedance functionality not implemented yet")

    curve_data, event_data = gev_return_curve(
        data,
        event_value,
        bootstrap_method=bootstrap_method,
        n_bootstraps=n_bootstraps,
        max_return_period=max_return_period,
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
