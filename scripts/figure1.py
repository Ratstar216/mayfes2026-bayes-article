from __future__ import annotations

import argparse
import csv
import os
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


def load_points(input_path: Path) -> tuple[list[float], list[float]]:
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    return xs, ys


def plot_points(xs: list[float], ys: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, color="tab:blue", s=45)
    ax.set_title("Plot of data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    default_input = scripts_dir / "data" / "data1_sample.csv"
    default_output = PROJECT_ROOT / "images" / "figure1.png"

    parser = argparse.ArgumentParser(
        description="Plot sampled points from scripts/data/data1_sample.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output image path.",
    )
    args = parser.parse_args()

    xs, ys = load_points(args.input)
    plot_points(xs, ys, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
