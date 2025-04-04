import numpy as np
import matplotlib.pyplot as plt

class BayesianAutoregressive:
    def __init__(self, order=1, prior_mean=None, prior_var=None):
        self.order = order
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.posterior_mean = None
        self.posterior_var = None
        self.coefficients = None

    def fit(self, y, iterations=1000):
        y = np.array(y)
        n = len(y)
        X = np.column_stack([y[i:n - self.order + i] for i in range(self.order)])
        y_target = y[self.order:]

        # Priors
        if self.prior_mean is None:
            self.prior_mean = np.zeros(self.order)
        if self.prior_var is None:
            self.prior_var = np.eye(self.order) * 10

        # Initial posterior
        sigma_sq = 1
        beta = np.zeros(self.order)

        beta_samples = []

        # Gibbs sampler
        for _ in range(iterations):
            # Update posterior variance and mean
            posterior_var_inv = np.linalg.inv(self.prior_var) + (X.T @ X) / sigma_sq
            self.posterior_var = np.linalg.inv(posterior_var_inv)
            self.posterior_mean = self.posterior_var @ (np.linalg.inv(self.prior_var) @ self.prior_mean + (X.T @ y_target) / sigma_sq)

            # Sample coefficients
            beta = np.random.multivariate_normal(self.posterior_mean, self.posterior_var)
            beta_samples.append(beta)

            # Update sigma_sq
            residuals = y_target - X @ beta
            shape = 0.5 * len(y_target)
            scale = 0.5 * np.sum(residuals ** 2)
            sigma_sq = 1 / np.random.gamma(shape, 1/scale)

        self.coefficients = np.mean(beta_samples[int(iterations/2):], axis=0)

    def predict(self, y_initial, steps=1):
        y_pred = list(y_initial[-self.order:])
        for _ in range(steps):
            next_pred = np.dot(self.coefficients, y_pred[-self.order:])
            y_pred.append(next_pred)
        return y_pred[-steps:]

    def plot_predictions(self, y_true, y_pred):
        plt.plot(y_true, label='Actual')
        plt.plot(range(len(y_true), len(y_true) + len(y_pred)), y_pred, label='Predicted', linestyle='--')
        plt.legend()
        plt.title('Bayesian AR Model Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()
