import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

class BAR:
    """
    Bayesian Autoregressive (BAR) model class using PyMC.

    This class models a univariate time series using a Bayesian AR(p) model,
    estimating posterior distributions over the AR coefficients and noise variance.

    Parameters
    ----------
    p : int
        Order of the autoregressive model (i.e., number of lags).
    priors : dict, optional
        Dictionary of prior specifications. Supports:
            - "phi": a callable returning a PyMC distribution (takes `shape=p`)
            - "sigma": a callable returning a PyMC distribution (takes no args)
            - OR: scalars "phi_mu", "phi_sigma", "sigma" to use defaults

    Attributes
    ----------
    model : pm.Model
        The PyMC model object.
    trace : arviz.InferenceData
        Posterior samples after fitting the model.
    y_fit : ndarray
        The original time series used during fitting.
    """

    ## Initialization
    def __init__(self, p=1, priors=None):
        """
        Initialize the BAR model.

        Parameters
        ----------
        p : int
            Order of the autoregressive model.
        priors : dict, optional
            Dictionary of prior specifications. Supports:
                - "phi": a callable returning a PyMC distribution (takes `shape=p`), default: N(mu = 0, sigma = 1).
                - "sigma": a callable returning a PyMC distribution (takes no args), default: Half-Normal(sigma = 1).
                - OR: scalars under "phi_mu", "phi_sigma", "sigma" to edit default priors parameters.
        """
        self.p = p
        self.y_fit = None
        self.model = None
        self.trace = None

        priors = priors or {}

        # Default phi prior
        if not callable(priors.get("phi")):
            phi_mu = priors.get("phi_mu", 0) # Returns value associated with "phi_mu" if it exists, otherwise returns 0.
            phi_sigma = priors.get("phi_sigma", 1)
            priors["phi"] = lambda shape: pm.Normal("phi", mu=phi_mu, sigma=phi_sigma, shape=shape)

        # Default sigma prior
        if not callable(priors.get("sigma")):
            sigma_scale = priors.get("sigma", 1)
            priors["sigma"] = lambda: pm.HalfNormal("sigma", sigma=sigma_scale)

        self.priors = priors



    ## Time Series + Prior Visualization
    def plot_series(self):
        """
        Plot the input time series used to fit the model.
        """
        if self.y_fit is None:
            raise RuntimeError("No data has been fit yet.")
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.y_fit, label="Input Time Series", color='black')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Observed Time Series")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_priors(self, samples=1000):
        """
        Plot the prior distributions for the AR coefficients and sigma.

        Can be run prior to model fitting.

        Parameters
        ----------
        samples : int
            Number of samples to draw from the prior distributions.
        """
        fig, axes = plt.subplots(1, self.p + 1, figsize=(4 * (self.p + 1), 4))

        with pm.Model():
            phi_rv = self.priors["phi"](shape=self.p)
            sigma_rv = self.priors["sigma"]()

            # Rebind priors with unique names for plotting context
            phi_temp = pm.Deterministic("phi_plot", phi_rv)
            sigma_temp = pm.Deterministic("sigma_plot", sigma_rv)

            prior_samples = pm.sample_prior_predictive(samples)

        # Extract and plot samples
        phi_samples = prior_samples.prior["phi_plot"].stack(sample=("chain", "draw")).values.T
        sigma_samples = prior_samples.prior["sigma_plot"].stack(sample=("chain", "draw")).values

        for i in range(self.p):
            axes[i].hist(phi_samples[:, i], bins=30, color="skyblue", edgecolor="gray", alpha=0.8)
            axes[i].set_title(f"Prior for φ[{i + 1}]")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")

        axes[-1].hist(sigma_samples, bins=30, color="lightcoral", edgecolor="gray", alpha=0.8)
        axes[-1].set_title("Prior for σ")
        axes[-1].set_xlabel("Value")
        axes[-1].set_ylabel("Density")

        plt.tight_layout()
        plt.show()



    ## Model fitting
    def fit(self, y, draws=1000, tune=1000, target_accept=0.9, random_seed=2102):
        """
        Fit the BAR(p) model to a univariate time series using MCMC.

        Parameters
        ----------
        y : array-like
            Univariate time series of shape (n_timesteps,).
        draws : int, optional
            Number of MCMC samples to draw (default is 1000).
        tune : int, optional
            Number of tuning steps for NUTS sampler (default is 1000).
        target_accept : float, optional
            Target acceptance rate for NUTS (default is 0.9).
        random_seed : int, optional
            Random seed for reproducibility (default is 2102).
        """
        y = np.asarray(y)
        if len(y) <= self.p:
            raise ValueError("Time series length must be greater than AR order p.")
        self.y_fit = y

        # Create lagged feature matrix - i.e. y[t-1]; y[t-2]; ...
        X = np.column_stack([y[i:-(self.p - i)] for i in range(self.p)])
        y_target = y[self.p:] # Reserve first p values for model fitting

        with pm.Model() as self.model:
            # Priors
            phi = self.priors["phi"](shape=self.p) # p priors
            sigma = self.priors["sigma"]()

            # Linear mean function - e.g. phi0 + phi1 * y[t-1] + ...
            mu = pm.math.dot(X, phi)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_target)

            # Posterior sampling
            self.trace = pm.sample(draws=draws, tune=tune,
                                   target_accept=target_accept,
                                   random_seed=random_seed)



    ## Model Summary + Diagnostics
    def summary(self):
        """
        Return summary statistics of posterior distributions.

        Returns
        -------
        summary_df : pandas.DataFrame
            Posterior summary from ArviZ.
        """
        if self.trace is None:
            raise RuntimeError("Model must be fit before calling summary().")
        return az.summary(self.trace)

    def plot_trace(self):
        """
        Plot MCMC trace and posterior histograms for all parameters.

        Returns
        -------
        matplotlib.figure.Figure
            Trace plot figure object.
        """
        if self.trace is None:
            raise RuntimeError("Model must be fit before calling plot_trace().")
        return az.plot_trace(self.trace)



    ## Forecasting
    def forecast(self, steps=10, credible_interval=0.9):
        """
        Forecast future values from the model.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.
        credible_interval : float
            Width of credible interval for uncertainty bands.

        Returns
        -------
        forecast_mean : ndarray
            Mean forecasted values.
        lower : ndarray
            Lower bound of the credible interval.
        upper : ndarray
            Upper bound of the credible interval.
        forecasts : ndarray
            All sampled forecast values (samples x steps).
        """
        if self.trace is None:
            raise RuntimeError("Fit the model before forecasting.")

        y_pred = np.asarray(self.y_fit).copy()
        forecasts = []

        for _ in range(steps):
            X_new = y_pred[-self.p:][::-1]
            samples = []
            for phi_sample, sigma_sample in zip(
                self.trace.posterior['phi'].stack(draws=("chain", "draw")).values.T,
                self.trace.posterior['sigma'].stack(draws=("chain", "draw")).values.flatten()
            ):
                mu = np.dot(X_new, phi_sample)
                samples.append(np.random.normal(mu, sigma_sample))
            forecast_dist = np.array(samples)
            y_pred = np.append(y_pred, np.mean(forecast_dist))
            forecasts.append(forecast_dist)

        forecasts = np.array(forecasts).T
        forecast_mean = forecasts.mean(axis=0)
        lower = np.percentile(forecasts, (1 - credible_interval) / 2 * 100, axis=0)
        upper = np.percentile(forecasts, (1 + credible_interval) / 2 * 100, axis=0)

        return forecast_mean, lower, upper, forecasts

    def plot_forecast(self, steps=10, credible_interval=0.9):
        """
        Plot forecasted values and uncertainty bands.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.
        credible_interval : float
            Width of credible interval for uncertainty bands.
        """
        forecast_mean, lower, upper, _ = self.forecast(steps, credible_interval)

        n_obs = len(self.y_fit)
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(n_obs), self.y_fit, label="Observed", color='black')
        plt.plot(np.arange(n_obs, n_obs + steps), forecast_mean, label="Forecast", color='blue')
        plt.fill_between(np.arange(n_obs, n_obs + steps), lower, upper,
                         color='blue', alpha=0.3, label=f"{int(credible_interval*100)}% CI")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"BAR({self.p}) Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()
