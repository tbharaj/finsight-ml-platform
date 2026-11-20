"""Command line interface: python -m finsight train / predict."""

import argparse
import json

from finsight.inference import load_model, predict_text
from finsight.training import run


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight financial-news sentiment research")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and evaluate the reference baseline")
    training.add_argument("--output-dir", default="artifacts")
    prediction = commands.add_parser("predict", help="Classify one financial headline")
    prediction.add_argument("--model", default="artifacts/sentiment_model.joblib")
    prediction.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "train":
        report = run(args.output_dir)
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(predict_text(load_model(args.model), args.text), indent=2))
