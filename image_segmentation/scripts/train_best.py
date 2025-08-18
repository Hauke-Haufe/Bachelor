import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from util.evaluation import Evaluation  # <-- anpassen, falls deine Klasse anders heißt oder woanders liegt

def main():
    parser = argparse.ArgumentParser(description="Train best config(s) from folds")
    parser.add_argument("folds_dir", type=str, help="Folder with container the folds with Paramterconfigs" )
    parser.add_argument("--fold", type=int, default=None, 
                        help="Fold-ID (wenn None, werden alle Folds trainiert)")
    args = parser.parse_args()

    trainer = Evaluation(args.folds_dir)
    if args.fold is not None:
        trainer.train_best_config_fold(args.fold)
    else:
        trainer.train_best_models()

    trainer.test_best_models()

if __name__ == "__main__":
    main()