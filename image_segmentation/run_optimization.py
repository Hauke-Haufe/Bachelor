from cross_validation import Crossvalidation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization.")
    parser.add_argument("path", help="Destination folder to save best parameters.")
    args = parser.parse_args()

    optimizer = Crossvalidation("dataset")
    optimizer.hyperparameter_optimization(args.path)