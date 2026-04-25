from __future__ import annotations

import os
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
cache_dir = scripts_dir.parent / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIOR_STD = 1.0
NOISE_STD = 0.2


def load_sample(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(sample_path, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def design_matrix(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def fit_bayesian_linear_regression(
    phi: np.ndarray,
    y: np.ndarray,
    prior_std: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = phi.shape[1]
    prior_variance = prior_std**2
    noise_variance = noise_std**2

    posterior_precision = (
        (phi.T @ phi) / noise_variance + np.eye(n_features) / prior_variance
    )
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ phi.T @ y / noise_variance

    return posterior_mean, posterior_covariance


def predictive_distribution(
    phi_plot: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictive_mean = phi_plot @ posterior_mean
    latent_variance = np.sum((phi_plot @ posterior_covariance) * phi_plot, axis=1)
    predictive_variance = latent_variance + noise_std**2
    predictive_std = np.sqrt(np.maximum(predictive_variance, 0.0))
    return predictive_mean, predictive_std


def main() -> None:
    sample_path = scripts_dir / "data" / "data1.csv"
    output_path = scripts_dir.parent / "images" / "bayes2_fit.png"

    x, y = load_sample(sample_path)
    phi = design_matrix(x)

    posterior_mean, posterior_covariance = fit_bayesian_linear_regression(
        phi=phi,
        y=y,
        prior_std=PRIOR_STD,
        noise_std=NOISE_STD,
    )

    # x_plot = np.linspace(x.min() - 0.2, x.max() + 0.2, 400)
    x_plot = np.linspace(-2.75, 2.75, 400)
    phi_plot = design_matrix(x_plot)
    predictive_mean, predictive_std = predictive_distribution(
        phi_plot=phi_plot,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        noise_std=NOISE_STD,
    )

    lower = predictive_mean - 2.0 * predictive_std
    upper = predictive_mean + 2.0 * predictive_std

    print("posterior mean:", posterior_mean)
    print("posterior covariance:")
    print(posterior_covariance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="tab:blue", label="sample data", zorder=3)
    ax.plot(
        x_plot,
        predictive_mean,
        color="tab:red",
        linewidth=2,
        label="Bayesian posterior mean",
    )
    ax.fill_between(
        x_plot,
        lower,
        upper,
        color="tab:red",
        alpha=0.2,
        label="predictive mean ± 2 std",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-4, 4)
    ax.set_title("Bayesian Linear Regression with Bases {1, x, x^2, x^3}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(output_path)


if __name__ == "__main__":
    main()
