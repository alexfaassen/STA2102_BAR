import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BayesianAR:
    def __init__(self, p=1, prior_mean=None, prior_var=None, sigma2=1.0):
        """
        Bayesian Autoregressive (BAR) model.

        Parameters:
        p : int
            Order of the autoregressive model.
        prior_mean : np.array
            Prior mean of AR parameters (size p).
        prior_var : np.array
            Prior covariance matrix of AR parameters (p x p).
        sigma2 : float
            Variance of the likelihood.
        """
        self.p = p
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(p)
        self.prior_var = prior_var if prior_var is not None else np.eye(p)
        self.sigma2 = sigma2
        self.posterior_mean = None
        self.posterior_var = None

    def fit(self, y):
        """
        Fit the BAR model using Bayesian updating.

        Parameters:
        y : np.array
            Time series data.
        """
        n = len(y)
        X = np.column_stack([y[i:n - self.p + i] for i in range(self.p)])
        y_target = y[self.p:]

        V_inv = np.linalg.inv(self.prior_var) + (X.T @ X) / self.sigma2
        V = np.linalg.inv(V_inv)
        m = V @ (np.linalg.inv(self.prior_var) @ self.prior_mean + X.T @ y_target / self.sigma2)

        self.posterior_mean = m
        self.posterior_var = V

    def forecast(self, y_last, steps=1):
        """
        Forecast future values using the posterior predictive distribution.

        Parameters:
        y_last : np.array
            Most recent p values of the time series.
        steps : int
            Number of steps to forecast.

        Returns:
        forecasts : np.array
            Forecasted values.
        """
        forecasts = []
        y_current = y_last.copy()

        for _ in range(steps):
            pred_mean = self.posterior_mean @ y_current[::-1]
            forecasts.append(pred_mean)

            y_current = np.roll(y_current, -1)
            y_current[-1] = pred_mean

        return np.array(forecasts)

    def plot_results(self, y, forecasts):
        """
        Plot the actual data and forecasts.

        Parameters:
        y : np.array
            Original time series data.
        forecasts : np.array
            Forecasted values.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y, label='Observed', marker='o')
        forecast_range = np.arange(len(y), len(y) + len(forecasts))
        plt.plot(forecast_range, forecasts, label='Forecast', marker='x', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Bayesian Autoregressive Model Forecast')
        plt.grid()
        plt.show()

# Example usage (to be placed in a separate test file or notebook):
# if __name__ == '__main__':
#     np.random.seed(0)
#     data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(scale=0.5, size=100)
#     bar_model = BayesianAR(p=2)
#     bar_model.fit(data)
#     forecasts = bar_model.forecast(data[-2:], steps=10)
#     bar_model.plot_results(data, forecasts)