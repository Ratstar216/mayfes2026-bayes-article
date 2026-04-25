from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
MPL_CACHE_DIR = LOCAL_CACHE_DIR / "matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class FrequentistResult:
    intercept: float
    slope: float
    sigma_squared: float


@dataclass
class BayesianResult:
    posterior_mean_intercept: float
    posterior_mean_slope: float
    posterior_var_intercept: float
    posterior_var_slope: float
    posterior_cov_intercept_slope: float


@dataclass
class PredictionSummary:
    x: float
    frequentist_mean: float
    bayesian_mean: float
    bayesian_std: float


def generate_data(
    n_samples: int,
    true_intercept: float,
    true_slope: float,
    noise_std: float,
    seed: int = 7,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    xs = [i / 2.0 for i in range(-n_samples // 2, n_samples // 2)]
    return [
        (x, true_intercept + true_slope * x + rng.gauss(0.0, noise_std))
        for x in xs
    ]


def fit_frequentist_linear_regression(
    data: list[tuple[float, float]],
) -> FrequentistResult:
    n = len(data)
    mean_x = sum(x for x, _ in data) / n
    mean_y = sum(y for _, y in data) / n

    sxx = sum((x - mean_x) ** 2 for x, _ in data)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in data)

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    residual_sum_squares = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in data
    )
    sigma_squared = residual_sum_squares / (n - 2)

    return FrequentistResult(
        intercept=intercept,
        slope=slope,
        sigma_squared=sigma_squared,
    )


def invert_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    return [[d / det, -b / det], [-c / det, a / det]]


def matmul_2x2_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]


def fit_bayesian_linear_regression(
    data: list[tuple[float, float]],
    noise_variance: float,
    prior_mean: tuple[float, float] = (0.0, 0.0),
    prior_variance: tuple[float, float] = (25.0, 25.0),
) -> BayesianResult:
    prior_precision = [
        [1.0 / prior_variance[0], 0.0],
        [0.0, 1.0 / prior_variance[1]],
    ]

    xtx = [[0.0, 0.0], [0.0, 0.0]]
    xty = [0.0, 0.0]

    for x, y in data:
        phi = [1.0, x]
        xtx[0][0] += phi[0] * phi[0]
        xtx[0][1] += phi[0] * phi[1]
        xtx[1][0] += phi[1] * phi[0]
        xtx[1][1] += phi[1] * phi[1]
        xty[0] += phi[0] * y
        xty[1] += phi[1] * y

    posterior_precision = [
        [
            prior_precision[0][0] + xtx[0][0] / noise_variance,
            prior_precision[0][1] + xtx[0][1] / noise_variance,
        ],
        [
            prior_precision[1][0] + xtx[1][0] / noise_variance,
            prior_precision[1][1] + xtx[1][1] / noise_variance,
        ],
    ]
    posterior_covariance = invert_2x2(posterior_precision)

    rhs = [
        prior_precision[0][0] * prior_mean[0] + xty[0] / noise_variance,
        prior_precision[1][1] * prior_mean[1] + xty[1] / noise_variance,
    ]
    posterior_mean = matmul_2x2_vec(posterior_covariance, rhs)

    return BayesianResult(
        posterior_mean_intercept=posterior_mean[0],
        posterior_mean_slope=posterior_mean[1],
        posterior_var_intercept=posterior_covariance[0][0],
        posterior_var_slope=posterior_covariance[1][1],
        posterior_cov_intercept_slope=posterior_covariance[0][1],
    )


def summarize_predictions(
    x_values: list[float],
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
) -> list[PredictionSummary]:
    summaries: list[PredictionSummary] = []
    for x in x_values:
        freq_mean = frequentist.intercept + frequentist.slope * x
        bayes_mean = (
            bayesian.posterior_mean_intercept + bayesian.posterior_mean_slope * x
        )
        bayes_var = (
            bayesian.posterior_var_intercept
            + (x**2) * bayesian.posterior_var_slope
            + 2.0 * x * bayesian.posterior_cov_intercept_slope
        )
        bayes_std = math.sqrt(max(bayes_var, 0.0))
        summaries.append(
            PredictionSummary(
                x=x,
                frequentist_mean=freq_mean,
                bayesian_mean=bayes_mean,
                bayesian_std=bayes_std,
            )
        )
    return summaries


def print_prediction_table(predictions: list[PredictionSummary]) -> None:
    print("\nPredictions")
    print("-" * 72)
    print(f"{'x':>8} {'Frequentist':>16} {'Bayesian mean':>16} {'Bayesian std':>16}")
    for prediction in predictions:
        print(
            f"{prediction.x:>8.2f}"
            f" {prediction.frequentist_mean:>16.4f}"
            f" {prediction.bayesian_mean:>16.4f}"
            f" {prediction.bayesian_std:>16.4f}"
        )


def bayesian_predictive_std(x: float, bayesian: BayesianResult) -> float:
    variance = (
        bayesian.posterior_var_intercept
        + (x**2) * bayesian.posterior_var_slope
        + 2.0 * x * bayesian.posterior_cov_intercept_slope
    )
    return math.sqrt(max(variance, 0.0))


def plot_results(
    data: list[tuple[float, float]],
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    output_path: Path,
) -> None:
    x_values = [x for x, _ in data]
    y_values = [y for _, y in data]
    x_min = min(x_values) - 1.0
    x_max = max(x_values) + 1.0
    grid = [x_min + i * (x_max - x_min) / 199 for i in range(200)]

    frequentist_line = [
        frequentist.intercept + frequentist.slope * x for x in grid
    ]
    bayesian_line = [
        bayesian.posterior_mean_intercept + bayesian.posterior_mean_slope * x
        for x in grid
    ]
    bayesian_std = [bayesian_predictive_std(x, bayesian) for x in grid]
    bayesian_upper = [
        mean + 2.0 * std for mean, std in zip(bayesian_line, bayesian_std)
    ]
    bayesian_lower = [
        mean - 2.0 * std for mean, std in zip(bayesian_line, bayesian_std)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(x_values, y_values, color="#1f2937", label="Observed data")
    axes[0].plot(grid, frequentist_line, color="#d97706", linewidth=2.5, label="Frequentist fit")
    axes[0].plot(grid, bayesian_line, color="#2563eb", linewidth=2.5, linestyle="--", label="Bayesian posterior mean")
    axes[0].fill_between(
        grid,
        bayesian_lower,
        bayesian_upper,
        color="#93c5fd",
        alpha=0.35,
        label="Bayesian 95% credible band",
    )
    axes[0].set_title("Fit Comparison")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    labels = ["Intercept", "Slope"]
    frequentist_values = [frequentist.intercept, frequentist.slope]
    bayesian_values = [
        bayesian.posterior_mean_intercept,
        bayesian.posterior_mean_slope,
    ]
    bayesian_errors = [
        math.sqrt(bayesian.posterior_var_intercept),
        math.sqrt(bayesian.posterior_var_slope),
    ]
    positions = [0, 1]
    width = 0.34

    axes[1].bar(
        [p - width / 2 for p in positions],
        frequentist_values,
        width=width,
        color="#f59e0b",
        label="Frequentist estimate",
    )
    axes[1].bar(
        [p + width / 2 for p in positions],
        bayesian_values,
        width=width,
        yerr=bayesian_errors,
        capsize=6,
        color="#3b82f6",
        label="Bayesian posterior mean ± 1 std",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_title("Parameter Comparison")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Frequentist vs Bayesian Linear Regression", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = generate_data(
        n_samples=20,
        true_intercept=1.5,
        true_slope=2.0,
        noise_std=1.2,
        seed=7,
    )

    frequentist = fit_frequentist_linear_regression(data)
    bayesian = fit_bayesian_linear_regression(
        data,
        noise_variance=frequentist.sigma_squared,
        prior_mean=(0.0, 0.0),
        prior_variance=(4.0, 4.0),
    )

    print("Frequentist vs Bayesian Linear Regression")
    print("=" * 72)
    print(f"Samples: {len(data)}")
    print(f"Estimated noise variance (from OLS residuals): {frequentist.sigma_squared:.4f}")

    print("\nFrequentist linear regression (ordinary least squares)")
    print("-" * 72)
    print(f"Intercept: {frequentist.intercept:.4f}")
    print(f"Slope:     {frequentist.slope:.4f}")

    print("\nBayesian linear regression (Gaussian prior + Gaussian likelihood)")
    print("-" * 72)
    print(f"Posterior mean intercept: {bayesian.posterior_mean_intercept:.4f}")
    print(f"Posterior mean slope:     {bayesian.posterior_mean_slope:.4f}")
    print(f"Posterior std intercept:  {math.sqrt(bayesian.posterior_var_intercept):.4f}")
    print(f"Posterior std slope:      {math.sqrt(bayesian.posterior_var_slope):.4f}")

    predictions = summarize_predictions(
        x_values=[-4.0, -2.0, 0.0, 2.0, 4.0],
        frequentist=frequentist,
        bayesian=bayesian,
    )
    print_prediction_table(predictions)

    output_path = PROJECT_ROOT / "regression_comparison.png"
    plot_results(data, frequentist, bayesian, output_path)

    print("\nExplanation")
    print("-" * 72)
    print("Frequentist regression gives one best-fit line.")
    print("Bayesian regression gives a distribution over plausible lines.")
    print("The blue shaded region shows uncertainty around the Bayesian fit.")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
