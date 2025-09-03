""" 
Script main

Main entry point for the full training/evaluation/prediction pipeline.

Usage:
    % python -m scripts.main --config configs/main.yaml
    % python -m scripts.main --config configs/main.yaml --default configs/default.yaml

Authors:
    Will Raftery

Date:
    31/07/2025
"""

import argparse
import logging

from omegaconf import OmegaConf

from src.utils.utils import load_config, get_experiment_directory

from scripts.train import main as train
from scripts.eval import main as evaluate
from scripts.predict import main as predict


def main(cfg=None, overwrite=False):
    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    experiment_dir = get_experiment_directory(cfg)
    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    train_results = train(cfg, overwrite=overwrite)
    logging.info(f"Training completed with results: {train_results}")

    eval_results = evaluate(cfg, overwrite=overwrite)
    logging.info(f"Evaluation completed with results: {eval_results}")

    infer_results = predict(cfg, overwrite=overwrite)
    logging.info(f"Inference completed with results: {infer_results}")

    return {
        "train_results": train_results,
        "eval_results": eval_results,
        "infer_results": infer_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full training and evaluation pipeline")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)

    main(cfg, overwrite=args.overwrite)
