"""General utility functions."""

import argparse
import re

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rv_continuous, genextreme, goodness_of_fit
from xarray import apply_ufunc
import xclim


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

    def _check_fit(self, theta, **kwargs):
        """Test goodness of fit and (if applicable) retry fit by generating an initial estimate."""
        data = kwargs.pop("data")

        if len(theta) == 3:
            # Stationary parameters
            shape, loc, scale = theta
            test_data = data
        else:
            # Non-stationary parameters (test middle 20% of data).
            # N.B. the non stationary parameters vary with the data distribution as a
            # function of the covariate, so we test a subset of data at a specific point.
            n = data.size
            t = n // 2  # Index of covariate (i.e., middle timestep in a timeseries)
            dt = n // 10  # 10% of datapoints to test for goodness of fit.

            # Subset data around midpoint.
            test_data = data.isel({data.dims[-1]: slice(max(0, t - dt), t + dt)})

            shape, loc, loc1, scale, scale1 = theta
            loc = loc + loc1 * t
            scale = scale + scale1 * t

        res = goodness_of_fit(
            genextreme, test_data, known_params=dict(c=shape, loc=loc, scale=scale)
        )

        # Accept the null distribution of the AD test.
        success = True if res.pvalue > 0.01 else False
        if not success:
            if not kwargs["generate_estimates"]:
                print("Data fit failed. Retrying with 'generate_estimates=True'.")
                kwargs["generate_estimates"] = True
                theta = self.fit(data, **kwargs)
            else:
                print("Data fit failed.")
        return theta

    def fit_stationary(self, data, user_estimates, generate_estimates):
        """Return estimates of shape, location and scale parameters using genextreme."""
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

    def _sum_finite(self, x):
        """Sum finite values and count the number of nonfinite values in a 1D array.

        This is a utility function used when evaluating the negative
        loglikelihood for a distribution and an array of samples. Stolen from
        https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_distn_infrastructure.py
        """
        finite_x = np.isfinite(x)
        bad_count = finite_x.size - np.count_nonzero(finite_x)
        return np.sum(x[finite_x]), bad_count

    def nllf(self, theta, data, x):
        """Penalised negative log-likelihood of GEV probability density function.

        A modified version of scipy.stats.genextremes.fit for fitting extreme value
        distribution parameters, in which the location and scale parameters can vary
        linearly with a covariate.
        A large, finite penalty (rather than infinite negative log-likelihood)
        is applied for observations beyond the support of the distribution.
        Suitable for stationary or non stationary distributions. The non-stationary
        parameters are returned if the input 'theta' incudes the varying
        location and scale parameters.

        Parameters
        ----------
        theta : tuple of floats
            Shape, location and scale parameters (shape, loc0, loc1, scale0, scale1).
        data, x: array-like
            Data and covariate to fit.

        Returns
        -------
        total : float
            The penalised negative likelihood function summed over all values of x.
        """
        if len(theta) == 5:
            # Non-stationary GEV parameters.
            shape, loc0, loc1, scale0, scale1 = theta
            loc = loc0 + loc1 * x
            scale = scale0 + scale1 * x

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
            f = np.where(
                Z > 0,
                np.log(scale)
                + (1 + 1 / shape) * np.ma.log(Z)
                + np.ma.power(Z, -1 / shape),
                np.inf,
            )

        f = np.where(scale > 0, f, np.inf)  # Scale parameter must be positive.

        # Sum function along all axes (where finite) & count infinite elements.
        total, n_bad = self._sum_finite(f)

        # Add large finite penalty instead of infinity (log of the largest useable float).
        total = total + n_bad * np.log(np.finfo(float).max) * 100
        return total

    def _fit(
        self,
        data,
        user_estimates=None,
        loc1=0,
        scale1=0,
        x=None,
        generate_estimates=False,
        stationary=True,
        check_fit=False,
        method="Nelder-Mead",
    ):
        """Return estimates of data distribution parameters.

        For stationary data, estimates the shape, location and scale parameters using
        scipy.stats.genextremes.fit(). For non-stationary data, also estimates the linear
        location and scale trend parameters using a penalised negative log-likelihood
        function with initial estimates based on the stationary fit.

        Parameters
        ----------
        data : xarray.dataArray
            Data as a function of a covariate (assumed to be on axis=-1).
        user estimates: list, default None
            Initial estimates of the shape, loc and scale parameters
        loc1, scale1 : float, default 0
            Initial estimates of the location and scale trend parameters
        x : array-like, default None (i.e., linear)
            The covariate (e.g., timesteps)
        stationary : bool, default True
            Fit as a stationary GEV using scipy.stats.genextremes.fit
        check_fit : bool, default False
            Test goodness of fit and attempt retry
        method : str, default 'Nelder-Mead'
            Method used for scipy.optimize.minimize

        Returns
        -------
        theta : tuple of floats
            Shape, location and scale parameters (and loc1, scale1 if applicable)

        Example
        -------
        '''
        n = 1000
        m = 1e-3
        x = np.arange(n)
        rvs = genextreme.rvs(-0.05, loc=3.2, scale=0.5, size=n, random_state=0)
        data = xr.DataArray(rvs + x * m, coords={'time': x})
        data.plot()

        fit_gev = ns_genextreme_gen().fit
        shape, loc, scale = fit_gev(data, stationary=True)
        shape, loc, loc1, scale, scale1 = fit_gev(data, stationary=False)

        data_2d = xr.concat([data, data + x * 1e-2], 'lat')
        theta = fit_gev(data_2d, stationary=False)
        print(theta.dims)
        ('lat', 'theta')
        '''
        """
        kwargs = locals()

        # Use genextremes to get stationary distribution parameters.
        shape, loc, scale = self.fit_stationary(
            data, user_estimates, generate_estimates
        )

        if stationary:
            theta = shape, loc, scale
        else:
            if x is None:
                x = np.arange(data.shape[-1], dtype=int)
            theta_i = -shape, loc, loc1, scale, scale1

            # Optimisation bounds (scale parameter must be non-negative).
            bounds = [(None, None), (None, None), (None, None), (0, None), (None, None)]

            # Minimise the negative log-likelihood function to get optimal theta.
            res = minimize(
                self.nllf, theta_i, args=(data, x), method=method, bounds=bounds
            )
            theta = res.x

            # Flip sign of shape for consistency with scipy.stats results.
            theta[0] *= -1

        if check_fit:
            kwargs.pop("self")
            theta = self._check_fit(theta, **kwargs)
        return np.array(theta)

    def fit(self, data, **kwargs):
        """xarray.apply_ufunc wrapper for ns_genextreme_gen._fit."""
        time_dim = data.dims[-1]  # Assumes the time/covariate is on the last axis.

        # Expected output of theta (3 parameters unless stationary=False is specified).
        n = 3 if kwargs.get("stationary", True) else 5

        theta = apply_ufunc(
            self._fit,
            data,
            input_core_dims=[[time_dim]],
            output_core_dims=[["theta"]],
            vectorize=True,
            dask="parallelized",
            kwargs=kwargs,
            dask_gufunc_kwargs=dict(
                output_dtypes=["float64"], output_sizes={"theta": n}
            ),
        )

        if len(data.shape) == 1:
            # If input is a 1D array, return a tuple of scalars instead of a data array.
            theta = tuple([i.item() for i in theta])
        return theta


fit_gev = ns_genextreme_gen().fit


def return_period(data, event, **kwargs):
    """Get return period for given event by fitting a GEV."""

    shape, loc, scale = fit_gev(data, **kwargs)
    return_period = genextreme.isf(
        event, shape, loc=loc, scale=scale
    )  # 1.0 / probability

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
    shape, loc, scale = fit_gev(data, generate_estimates=True)

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
