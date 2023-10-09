"""General utility functions."""

import argparse
import re

import numpy as np
from scipy.optimize import minimize
import scipy.stats
from scipy.stats import rv_continuous
from scipy.stats import genextreme as gev
from scipy.stats._constants import _LOGXMAX
from scipy.stats._distn_infrastructure import _sum_finite
import xarray as xr
#import xclim




class store_dict(argparse.Action):
    """An argparse action for parsing a command line argument as a dictionary.

    Examples
    --------
    precip=mm/day becomes {'precip': 'mm/day'}
    ensemble=1:5 becomes {'ensemble': slice(1, 5)}
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, val = value.split("=")
            if ":" in val:
                start, end = val.split(":")
                try:
                    start = int(start)
                    end = int(end)
                except ValueError:
                    pass
                val = slice(start, end)
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = val


def convert_units(da, target_units):
    """Convert units.

    Parameters
    ----------
    da : xarray DataArray
        Input array containing a units attribute
    target_units : str
        Units to convert to

    Returns
    -------
    da : xarray DataArray
       Array with converted units
    """

    xclim_unit_check = {
        "deg_k": "degK",
        "kg/m2/s": "kg m-2 s-1",
    }

    if da.attrs["units"] in xclim_unit_check:
        da.attrs["units"] = xclim_unit_check[da.units]

    try:
        da = xclim.units.convert_units_to(da, target_units)
    except Exception as e:
        in_precip_kg = da.attrs["units"] == "kg m-2 s-1"
        out_precip_mm = target_units in ["mm d-1", "mm day-1"]
        if in_precip_kg and out_precip_mm:
            da = da * 86400
            da.attrs["units"] = target_units
        else:
            raise e

    return da


def date_pair_to_time_slice(date_list):
    """Convert two dates to a time slice object.

    Parameters
    ----------
    date_list : list or tuple
        Start and end date in YYYY-MM-DD format

    Returns
    -------
    time_slice : slice
        Slice from start to end date
    """

    assert len(date_list) == 2
    start_date, end_date = date_list

    date_pattern = "([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})"
    assert re.search(date_pattern, start_date), "Start date not in YYYY-MM-DD format"
    assert re.search(date_pattern, end_date), "End date not in YYYY-MM-DD format"

    time_slice = slice(start_date, end_date)

    return time_slice


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


class ns_genextreme_gen(rv_continuous):
    """Extreme value distributions (stationary or non-stationary)."""

    def fit_stationary(self, data, user_estimates, generate_estimates):
        """Return estimates of shape, location and scale parameters using genextreme."""
        if user_estimates:
            shape, loc, scale = user_estimates
            shape, loc, scale = gev.fit(data, shape, loc=loc, scale=scale)

        elif generate_estimates:
            # Generate initial estimates using a data subset (useful for large datasets).
            shape, loc, scale = gev.fit(data[::2])
            shape, loc, scale = gev.fit(data, shape, loc=loc, scale=scale)
        else:
            shape, loc, scale = gev.fit(data)
        return shape, loc, scale

    def nllf(self, theta, data, times):
        """Penalised negative log-likelihood of GEV probability density function.

        A modified version of scipy.stats.genextremes.fit for fitting extreme value
        distribution parameters, in which the location and scale parameters can vary
        linearly with a covariate. The non-stationary parameters will be returned only
        if the input 'theta' incudes the time-varying location and scale parameters.
        A large, finite penalty (rather than infinite negative log-likelihood)
        is applied for observations beyond the support of the distribution.

        Parameters
        ----------
        theta : tuple of floats
            Shape, location and scale parameters (shape, loc0, loc1, scale0, scale1).
        data, times : array_like
            Data time series and indexes of covariates.

        Returns
        -------
        total : float
            The sum of the penalised negative likelihood function.
        """

        if len(theta) == 5:
            # Non-stationary GEV parameters.
            shape, loc0, loc1, scale0, scale1 = theta
            loc = loc0 + loc1 * times
            scale = scale0 + scale1 * times

        else:
            # Stationary GEV parameters.
            shape, loc, scale = theta

        s = (data - loc) / scale

        # Calculate the NLLF (type 1 or types 2-3 extreme value distributions).
        if shape == 0:
            f = np.log(scale) + s + np.exp(-s)

        else:
            Z = 1 + shape * s
            # NLLF at points where the data is supported by the distribution parameters.
            # (N.B. the NLLF is not finite when the shape is nonzero and Z is negative
            # because the PDF is zero (log(0)=inf) outside of these bounds).
            f = np.where(Z > 0, (np.log(scale) + (1 + 1 / shape) * np.ma.log(Z) +
                                 np.ma.power(Z, -1 / shape)), np.inf)

        f = np.where(scale > 0, f, np.inf)  # Scale parameter must be positive.

        # Sum function along all axes (where finite) & count infinite elements.
        total, n_bad = _sum_finite(f)

        # Add large finite penalty instead of infinity (log of the largest useable float).
        total = total + n_bad * _LOGXMAX * 100
        return total

    def fit(self, data, user_estimates=[], loc1=0, scale1=0, generate_estimates=False,
            stationary=True, method='Nelder-Mead'):
        """Return estimates of data distribution parameters and their trend (if applicable).

        For stationary data, estimates the shape, location and scale parameters using
        scipy.stats.genextremes.fit(). For non-stationary data, also estimates the linear
        location and scale trend parameters using a penalised negative log-likelihood
        function with initial estimates based on the stationary fit.

        Parameters
        ----------
        data : array_like
            Data timeseries.
        user estimates: list, optional
            Initial estimates of the shape, loc and scale parameters.
        loc1, scale1 : float, optional
            Initial estimates of the location and scale trend parameters. Defaults to 0.
        stationary : bool, optional
            Fit as a stationary GEV using scipy.stats.genextremes.fit. Defaults to True.
        method : str, optional
            Method used for scipy.optimize.minimize. Defaults to 'Nelder-Mead'.

        Returns
        -------
        theta : tuple of floats
            Shape, location and scale parameters (and loc1 and scale1 if applicable)

        Example
        -------
        '''
        ns_genextreme = ns_genextreme_gen()
        data = scipy.stats.genextreme.rvs(0.8, loc=3.2, scale=0.5, size=500, random_state=0)
        shape, loc, scale = ns_genextreme.fit(data, stationary=True)
        shape, loc, loc1, scale, scale1 = ns_genextreme.fit(data, stationary=False)
        '''
        """

        # Use genextremes to get stationary distribution parameters.
        shape, loc, scale = self.fit_stationary(data, user_estimates, generate_estimates)

        if stationary:
            theta = shape, loc, scale
        else:
            times = np.arange(data.shape[-1], dtype=int)
            theta_i = shape, loc, loc1, scale, scale1

            # Optimisation bounds (scale parameter must be non-negative).
            bounds = [(None, None), (None, None), (None, None),
                      (0, None), (None, None)]

            # Minimise the negative log-likelihood function to get optimal theta.
            res = minimize(self.nllf, theta_i, args=(data, times),
                           method=method, bounds=bounds)
            theta = res.x

        return theta


fit_gev = ns_genextreme_gen().fit


def return_period(data, event, **kwargs):
    """Get return period for given event by fitting a GEV."""

    shape, loc, scale = fit_gev(data, **kwargs)
    return_period = gev.isf(event, shape, loc=loc, scale=scale)  # 1.0 / probability

    return return_period


def gev_return_curve(
    data, event_value, bootstrap_method="non-parametric", n_bootstraps=1000
):
    """Return x and y data for a GEV return period curve.

    Parameters
    ----------
    data : xarray DataArray
    event_value : float
        Magnitude of event of interest
    bootstrap_method : {'parametric', 'non-parametric'}, default 'non-parametric'
    n_bootstraps : int, default 1000

    """

    # GEV fit to data
    shape, loc, scale = fit_gev(data, generate_estimates=True)

    curve_return_periods = np.logspace(0, 4, num=10000)
    curve_probabilities = 1.0 / curve_return_periods
    curve_values = gev.isf(curve_probabilities, shape, loc, scale)

    event_probability = gev.sf(event_value, shape, loc=loc, scale=scale)
    event_return_period = 1.0 / event_probability

    # Bootstrapping for confidence interval
    boot_values = curve_values
    boot_event_return_periods = []
    for i in range(n_bootstraps):
        if bootstrap_method == "parametric":
            boot_data = gev.rvs(shape, loc=loc, scale=scale, size=len(data))
        elif bootstrap_method == "non-parametric":
            boot_data = np.random.choice(data, size=data.shape, replace=True)
        boot_shape, boot_loc, boot_scale = fit_gev(boot_data, generate_estimates=True)

        boot_value = gev.isf(curve_probabilities, boot_shape, boot_loc, boot_scale)
        boot_values = np.vstack((boot_values, boot_value))

        boot_event_probability = gev.sf(
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
    ax, data, event_value, bootstrap_method="parametric", n_bootstraps=1000, ylabel=None
):
    """Plot a single return period curve.

    Parameters
    ----------
    data : xarray DataArray
    """

    curve_data, event_data = gev_return_curve(
        data,
        event_value,
        bootstrap_method=bootstrap_method,
        n_bootstraps=n_bootstraps,
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
        curve_return_periods, curve_values, color="tab:blue", label="GEV fit to data"
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
    print(f"{event_return_period:.0f} year return period")
    print(
        f"95% CI: {event_return_period_lower_ci:.0f}-{event_return_period_upper_ci:.0f} years"
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

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[3], handles[0], handles[1], handles[2]]
    labels = [labels[3], labels[0], labels[1], labels[2]]
    ax.legend(handles, labels, loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("return period (years)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ylim = ax.get_ylim()
    ax.set_ylim([50, ylim[-1]])
    ax.grid()
