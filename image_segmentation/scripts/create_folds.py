import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from util.cross_validation import Training 

def main():
    parser = argparse.ArgumentParser(description="Create dataset folds for cross-validation.")
    parser.add_argument("dataset", type=str, help="The Folder with the dataset runs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds to create")
    parser.add_argument("--option", type=str, default="folds_in_run", help="Folding option")
    args = parser.parse_args()

    trainer = Training(args.dataset)
    trainer.create_folds(k_folds=args.k_folds, option=args.option)

    print(f"Folds created with {args.k_folds} folds (option='{args.option}')")

if __name__ == "__main__":
    main()