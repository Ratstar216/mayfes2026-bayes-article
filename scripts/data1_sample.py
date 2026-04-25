from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def sample_rows(
    input_path: Path,
    n_samples: int = 5,
    seed: int = 7,
) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if n_samples > len(rows):
        raise ValueError(
            f"Requested {n_samples} samples, but only {len(rows)} rows are available."
        )

    rng = random.Random(seed)
    return rng.sample(rows, n_samples)


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    default_input = scripts_dir / "data" / "data1.csv"
    default_output = scripts_dir / "data" / "data1_sample.csv"

    parser = argparse.ArgumentParser(
        description="Sample rows from scripts/data/data1.csv."
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
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    args = parser.parse_args()

    sampled_rows = sample_rows(
        input_path=args.input,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(args.output)


if __name__ == "__main__":
    main()
