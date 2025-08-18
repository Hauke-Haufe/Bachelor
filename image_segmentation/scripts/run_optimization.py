import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.cross_validation import Training
import argparse


if __name__ == "__main__":
    """
    Command-line entry point for hyperparameter optimization on a sepcific Fold.

    Usage
    -----
    CUDA_VISIBLE_DEVICES=<device_id> python run_optimization.py <path>

    Parameters
    ----------
    path : str
        Destination folder (fold directory) where the Optuna study,
        configs, and results will be stored.
    """

    parser = argparse.ArgumentParser(description="Runs hyperparameter optimization.")
    parser.add_argument("path", help="Fold Path where eval.txt and train.txt live.")
    parser.add_argument("--num_iterations", type=int, default=200, help="Number of Itertion the Hyperparamter Optimizer does")
    args = parser.parse_args()

    optimizer =Training("dataset")
    optimizer.hyperparameter_optimization(args.path, args.num_iterations)