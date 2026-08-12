"""CLI for the learning workflow."""
import argparse
from .data import make_data, save_data


def main() -> None:
    """Generate data or train a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate-data", "train"])
    parser.add_argument("--drift", default="normal", choices=["normal", "feature_drift", "concept_drift"])
    args = parser.parse_args()
    if args.command == "generate-data":
        save_data(make_data(drift=args.drift))
    else:
        from .train import train
        print(f"ROC-AUC: {train():.4f}")


if __name__ == "__main__":
    main()
