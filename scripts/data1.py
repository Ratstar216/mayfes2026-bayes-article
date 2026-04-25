from __future__ import annotations

import argparse
import random
from pathlib import Path


def sample_points(
    n_samples: int = 100,
    x_min: float = -2.75,
    x_max: float = 2.75,
    noise_std: float = 0.2,
    seed: int = 7,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    points: list[tuple[float, float]] = []

    for _ in range(n_samples):
        x = rng.uniform(x_min, x_max)
        noise = rng.gauss(0.0, noise_std)
        y = x - (x**3) / 6.0 + noise
        points.append((x, y))

    return points


def main() -> None:
    project_root = Path(__file__).resolve().parent
    default_output = project_root / "data" / "data1.csv"

    parser = argparse.ArgumentParser(
        description="Generate sampled points from y = x - x^3 / 6 + noise."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of sampled points.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.2,
        help="Standard deviation of Gaussian noise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    points = sample_points(
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    with args.output.open("w", encoding="utf-8") as file:
        file.write("x,y\n")
        for x, y in points:
            file.write(f"{x:.6f},{y:.6f}\n")

    print(args.output)


if __name__ == "__main__":
    main()
