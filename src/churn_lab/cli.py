"""Run from the repository root after `pip install -e .`."""

import argparse
from pathlib import Path

import pandas as pd

from churn_lab.data import ROOT, feature_frame
from churn_lab.modeling import load_bundle, probabilities, train_project


def main():
    parser = argparse.ArgumentParser(description="Telco Churn Lab — Ainur Ali")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Reproduce the full experiment")
    train.add_argument("--trials", type=int, default=16)
    train.add_argument("--bootstrap", type=int, default=400)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--out", type=Path, default=ROOT)
    score = commands.add_parser("score", help="Score a CSV with the verified local model")
    score.add_argument("input", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train_project(
            out=args.out.resolve(),
            trials=args.trials,
            bootstrap=args.bootstrap,
            seed=args.seed,
            threads=args.threads,
        )
        from churn_lab.reporting import write_reports

        write_reports(args.out.resolve())
    else:
        bundle = load_bundle()
        raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        frame = feature_frame(raw, training=False)
        probability = probabilities(bundle, frame, bundle["champion"])
        result = pd.DataFrame(
            {
                "customerID": frame.customerID,
                "churn_probability": probability,
                "MonthlyCharges": frame.MonthlyCharges,
            }
        )
        result = result.sort_values(["churn_probability", "customerID"], ascending=[False, True])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Scored {len(result):,} customers with {bundle['champion']}: {args.output}")


if __name__ == "__main__":
    main()
