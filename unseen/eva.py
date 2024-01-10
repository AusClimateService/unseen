"""Extreme value analysis functions."""

import warnings

from matplotlib.dates import date2num
import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme, goodness_of_fit
from xarray import apply_ufunc


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
    user_estimates=None,
    loc1=0,
    scale1=0,
    covariate="time",
    core_dim="time",
    stationary=True,
    check_fit=True,
    alpha=0.05,
    generate_estimates=False,
):
    """Estimate parameters for stationary or nonstationary data distributions.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional data array to fit
    user estimates: list, default None
        Initial estimates of the shape, loc and scale parameters
    loc1, scale1 : float, default 0
        Initial estimates of the location and scale trend parameters
    covariate : array-like or string, default "time" (required if 'stationary=False')
        A non-stationary covariate array or coordinate name.
    core_dim : str, default "time"
        Name of time/sample dimension in 'data'
    stationary : bool, default True
        Fit as a stationary GEV using 'scipy.stats.genextremes.fit'
    check_fit : bool, default False
        Test goodness of fit and attempt retry
    alpha : float, default 0.05
        Goodness of fit p-value threshold
    generate_estimates : bool, default False
        Generate initial parameter guesses using a data subset

    Returns
    -------
    theta : tuple
        Shape, location and scale parameters (and loc1, scale1 if applicable)

    Notes
    -----
    - For stationary data the shape, location and scale parameters are
    estimated using 'scipy.stats.genextremes.fit()'.
    - For non-stationary data, the linear location and scale trend
    parameters are estimated using a penalised negative log-likelihood
    function with initial estimates based on the stationary fit.
    - The distribution fit is considered good if the p-value is above
    the 99th percent level (i.e., accept the null hypothesis).
    - If the parameters fail the goodness of fit test, it will attempt
    to fit the data again by generating an initial guess - if that
    hasn't already been tried.
    - If data is a stacked forecast ensemble, the covariate will need to be
    stacked in the same way.

    Todo
    ----
    - Test if data is trended before nonstationary fit
    """
    kwargs = locals()  # Function inputs

    def fit_stationary_gev(data, user_estimates, generate_estimates):
        """Estimate stationary shape, location and scale parameters."""
        if user_estimates:
            shape, loc, scale = user_estimates
            shape, loc, scale = genextreme.fit(data, shape, loc=loc, scale=scale)

        elif generate_estimates:
            # Generate initial estimates using a data subset (useful for large datasets).
            shape, loc, scale = genextreme.fit(data[::2])
            shape, loc, scale = genextreme.fit(data, shape, loc=loc, scale=scale)
        else:
            shape, loc, scale = genextreme.fit(data)
        return shape, loc, scale

    def penalised_sum(x):
        """Sum finite values in a 1D array and add nonfinite penalties.

        This is a utility function used when evaluating the negative
        loglikelihood for a distribution and an array of samples.
        Adapted/stolen from:
        https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_distn_infrastructure.py
        """
        finite_x = np.isfinite(x)
        bad_count = finite_x.size - np.count_nonzero(finite_x)

        total = np.sum(x[finite_x])
        penalty = bad_count * np.log(np.finfo(float).max) * 100

        return total + penalty

    def nllf(theta, data, covariate):
        """Penalised negative log-likelihood function.

        Parameters
        ----------
        theta : tuple of floats
            Distribution parameters (can be stationary or non-stationary)
            or
        data, covariate: array-like
            Data to fit and covariate (e.g., timesteps).

        Returns
        -------
        total : float
            The penalised NLLF summed over all values of x.

        Notes
        -----
        - A modified version of scipy.stats.genextremes.fit for fitting
        extreme value distribution parameters, in which the location and
        scale parameters can vary linearly with a covariate.
        - Suitable for stationary or non-stationary distributions:
            - theta = (shape, loc, scale)
            - theta = (shape, loc, loc1, scale, scale1)
        - The non-stationary parameters are returned if the input theta
        incudes the varying location and scale parameters.
        - A large finite penalty (instead of infinity) is applied for
        observations beyond the support of the distribution.
        - The NLLF is not finite when the shape is nonzero and Z is
        negative because the PDF is zero (i.e., log(0)=inf)).
        """
        if len(theta) == 5:
            # Non-stationary GEV parameters.
            shape, loc0, loc1, scale0, scale1 = theta
            loc = loc0 + loc1 * covariate
            scale = scale0 + scale1 * covariate

        else:
            # Stationary GEV parameters.
            shape, loc, scale = theta

        s = (data - loc) / scale

        # Calculate the NLLF (type 1 or types 2-3 extreme value distributions).
        if shape == 0:
            f = np.log(scale) + s + np.exp(-s)

        else:
            Z = 1 + shape * s
            # NLLF where data is supported by the parameters (see notes).
            f = np.where(
                Z > 0,
                np.log(scale)
                + (1 + 1 / shape) * np.ma.log(Z)
                + np.ma.power(Z, -1 / shape),
                np.inf,
            )

        f = np.where(scale > 0, f, np.inf)  # Scale parameter must be positive.

        # Sum function along all axes (where finite) & add penalty for each infinite element.
        total = penalised_sum(f)
        return total

    def _fit(
        data,
        user_estimates,
        loc1,
        scale1,
        covariate,
        core_dim,
        stationary,
        check_fit,
        alpha,
        generate_estimates,
    ):
        """Estimate distribution parameters."""
        # Use genextremes to get stationary distribution parameters.
        shape, loc, scale = fit_stationary_gev(data, user_estimates, generate_estimates)

        if stationary:
            theta = shape, loc, scale
        else:
            # Initial parameter guesses.
            theta_i = -shape, loc, loc1, scale, scale1

            # Optimisation bounds (scale parameter must be non-negative).
            bounds = [(None, None)] * 5
            bounds[3] = (0, None)

            # Minimise the negative log-likelihood function to get optimal theta.
            res = minimize(
                nllf,
                theta_i,
                args=(data, covariate),
                method="Nelder-Mead",
                bounds=bounds,
            )
            theta = res.x

            # Restore 'shape' sign for consistency with scipy.stats results.
            theta[0] *= -1

        theta = np.array([i for i in theta], dtype="float64")

        if check_fit:
            pvalue = check_gev_fit(
                data, theta, covariate=kwargs["covariate"], core_dim=kwargs["core_dim"]
            )

            # Accept the null distribution of the AD test.
            success = True if np.all(pvalue >= alpha) else False
            if not success:
                if not kwargs["generate_estimates"]:
                    warnings.warn(
                        "Data fit failed. Retrying with 'generate_estimates=True'."
                    )
                    kwargs["generate_estimates"] = True  # Also breaks loop
                    theta = fit(data, **kwargs)
                else:
                    warnings.warn("Data fit failed.")
                    shape, loc, scale
        return theta

    def fit(data, **kwargs):
        """xarray.apply_ufunc wrapper for _fit."""
        stationary = kwargs["stationary"]
        core_dim = kwargs["core_dim"]

        # Expected output of theta
        n = 3 if stationary else 5

        theta = apply_ufunc(
            _fit,
            data,
            input_core_dims=[[core_dim]],
            output_core_dims=[["theta"]],
            vectorize=True,
            dask="parallelized",
            kwargs=kwargs,
            output_dtypes=["float64"],
            dask_gufunc_kwargs=dict(output_sizes={"theta": n}),
        )
        return theta

    data = kwargs.pop("data")

    if not stationary and isinstance(covariate, str):
        covariate = data[covariate]
        if covariate.dtype.kind not in set("buifc"):
            # Convert dates to numbers
            covariate = date2num(covariate)
        kwargs["covariate"] = covariate

    theta = fit(data, **kwargs)

    # Return a tuple of scalars instead of a 1D data array
    if len(data.shape) == 1:
        theta = np.array([i for i in theta], dtype="float64")
    return theta


def check_gev_fit(data, theta, covariate=None, core_dim="time"):
    """Test stationary distribution goodness of fit.

    Parameters
    ----------
    data, x: array-like
        Data and covariate to fit.
    theta : tuple of floats
        Shape, location and scale parameters.
    covariate : array-like, optional
        Non-stationary covariate.
    core_dim : str, default is "time"
        Data dimension to test over.

    Returns
    -------
    pvalue : scipy.stats._fit.GoodnessOfFitResult.pvalue
        Goodness of fit pvalue.

    Notes
    -----
    - The nonstationary parameters vary with the data distribution as a function
    of the covariate, so we test a subset of data at a specific point.
    - For nonstationary distributions, the parameters vary as a function of the
    covariate the stationary fit is only compared to a subset of data near
    the middle of the timeseries.

    Todo
    ----
    - Add goodness of fit and trend significance test for nonstationary parameters
    """

    def _goodness_of_fit(data, theta, covariate=None):
        assert len(theta) in [3, 5]
        if len(theta) == 3:
            # Stationary parameters
            shape, loc, scale = theta
            test_data = data

        elif len(theta) == 5:
            # Non-stationary parameters (test middle 20% of data).

            n = data.size
            # Index of covariate (i.e., middle timestep in a timeseries)
            if covariate is None:
                t = n // 2
            else:
                t = int(np.median(np.unique(covariate)))
            dt = n // 10  # Subset 2*10% of data.

            # Subset data around midpoint.
            test_data = data[t - dt : t + dt]

            shape, loc, loc1, scale, scale1 = theta
            loc = loc + loc1 * t
            scale = scale + scale1 * t

        res = goodness_of_fit(
            genextreme,
            test_data,
            known_params=dict(c=shape, loc=loc, scale=scale),
        )
        return res.pvalue

    pvalue = apply_ufunc(
        _goodness_of_fit,
        data,
        theta,
        input_core_dims=[[core_dim], ["theta"]],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(covariate=covariate),
        dask_gufunc_kwargs=dict(meta=(np.ndarray(1, float),)),
    )
    return pvalue


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
    user_estimates=None,
):
    """Return x and y data for a GEV return period curve.

    Parameters
    ----------
    data : xarray DataArray
    event_value : float
        Magnitude of event of interest
    bootstrap_method : {'parametric', 'non-parametric'}, default "non-parametric"
    n_bootstraps : int, default 1000
    max_return_period : float, default 4
        The maximum return period is 10^{max_return_period}
    user_estimates: list, default None
        Initial estimates of the shape, loc and scale parameters
    """

    # GEV fit to data
    if user_estimates:
        shape, loc, scale = fit_gev(data, user_estimates=user_estimates, stationary=True)
    else:
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
    user_estimates=None,
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
    user_estimates: list, default None
        Initial estimates of the shape, loc and scale parameters
    """

    if direction == "deceedance":
        ValueError("Deceedance functionality not implemented yet")

    curve_data, event_data = gev_return_curve(
        data,
        event_value,
        bootstrap_method=bootstrap_method,
        n_bootstraps=n_bootstraps,
        max_return_period=max_return_period,
        user_estimates=user_estimates,
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
