import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

class BAR:
    def __init__(self, p=1, priors=None):
        """
        Initialize the BAR model.

        Parameters:
        - p (int): Order of the autoregressive model.
        - priors (dict): Optional. Custom prior configurations.
        """
        self.p = p
        self.model = None
        self.trace = None
        self.priors = priors or {}

    def fit(self, y, draws=1000, tune=1000, target_accept=0.9, random_seed=42):
        """
        Fit the BAR(p) model using MCMC sampling.

        Parameters:
        - y (array-like): 1D time series.
        - draws (int): Number of posterior samples to draw.
        - tune (int): Number of tuning steps.
        - target_accept (float): Target acceptance rate for HMC.
        - random_seed (int): Seed for reproducibility.
        """
        y = np.asarray(y)
        if len(y) <= self.p:
            raise ValueError("Time series length must be greater than AR order p.")

        X = np.column_stack([y[i:-(self.p - i)] for i in range(self.p)])
        y_target = y[self.p:]

        with pm.Model() as self.model:
            # Priors for AR coefficients
            phi = pm.Normal("phi", mu=self.priors.get("phi_mu", 0),
                            sigma=self.priors.get("phi_sigma", 1),
                            shape=self.p)
            
            # Prior for noise std deviation
            sigma = pm.HalfNormal("sigma", sigma=self.priors.get("sigma", 1))

            # Mean of the AR process
            mu = pm.math.dot(X, phi)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_target)

            # Sample from posterior
            self.trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed)

    def summary(self):
        """Return a summary of the posterior distributions."""
        if self.trace is None:
            raise RuntimeError("Model must be fit before summary.")
        return az.summary(self.trace)

    def plot_trace(self):
        """Plot MCMC trace plots."""
        if self.trace is None:
            raise RuntimeError("Model must be fit before plotting.")
        return az.plot_trace(self.trace)

    def sample_posterior_predictive(self, y):
        """
        Generate posterior predictive samples based on observed input.

        Parameters:
        - y (array-like): The observed series used in fitting.
        """
        if self.trace is None or self.model is None:
            raise RuntimeError("Fit the model before sampling posterior predictive.")

        return pm.sample_posterior_predictive(self.trace, model=self.model)
    
    def plot_forecast(self, steps=10, credible_interval=0.9):
        if self.trace is None:
            raise RuntimeError("Fit the model before forecasting.")

        y_pred = np.asarray(self.y_fit).copy()
        forecasts = []

        for _ in range(steps):
            X_new = y_pred[-self.p:][::-1]  # Last p values
            samples = []
            for phi_sample, sigma_sample in zip(self.trace.posterior['phi'].stack(draws=("chain", "draw")).values.T,
                                                 self.trace.posterior['sigma'].stack(draws=("chain", "draw")).values.flatten()):
                mu = np.dot(X_new, phi_sample)
                samples.append(np.random.normal(mu, sigma_sample))
            forecast_dist = np.array(samples)
            y_pred = np.append(y_pred, np.mean(forecast_dist))
            forecasts.append(forecast_dist)

        forecasts = np.array(forecasts).T  # shape: (samples, steps)
        forecast_mean = forecasts.mean(axis=0)
        lower = np.percentile(forecasts, (1 - credible_interval) / 2 * 100, axis=0)
        upper = np.percentile(forecasts, (1 + credible_interval) / 2 * 100, axis=0)

        # Plot
        plt.figure(figsize=(10, 5))
        n_obs = len(self.y_fit)
        plt.plot(np.arange(n_obs), self.y_fit, label="Observed", color='black')
        plt.plot(np.arange(n_obs, n_obs + steps), forecast_mean, label="Forecast", color='blue')
        plt.fill_between(np.arange(n_obs, n_obs + steps), lower, upper, color='blue', alpha=0.3, label=f"{int(credible_interval*100)}% CI")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"BAR({self.p}) Forecast")
        plt.tight_layout()
        plt.show()

