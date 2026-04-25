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


def load_sample(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(sample_path, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def design_matrix(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def main() -> None:
    sample_path = scripts_dir / "data" / "data1_sample.csv"
    output_path = scripts_dir.parent / "images" / "frequentist1_fit.png"

    x, y = load_sample(sample_path)

    phi = design_matrix(x)
    coeffs, *_ = np.linalg.lstsq(phi, y, rcond=None)

    print(coeffs)  # [a0, a1, a2, a3]

    # x_plot = np.linspace(x.min() - 0.2, x.max() + 0.2, 400)
    x_plot = np.linspace(-2.75, 2.75, 400)
    y_plot = design_matrix(x_plot) @ coeffs

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="tab:blue", label="sample data", zorder=3)
    ax.plot(x_plot, y_plot, color="tab:red", label="least-squares cubic fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-4, 4)
    ax.set_title("Least-Squares Fit with Bases {1, x, x^2, x^3}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(output_path)


if __name__ == "__main__":
    main()
