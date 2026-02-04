"""
Script eval

Handles model testing/evaluation.

Usage:
    $ python -m scripts.eval --config configs/eval.yaml
    $ python -m scripts.eval --config configs/eval.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import argparse
from pathlib import Path
import logging
import csv

from omegaconf import OmegaConf

from typing import Union, Optional, Callable

from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.models import setup_model

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def save_results(save_dir, results_dict):
    """
    Save test results (dict of metrics) to a CSV file
    """
    test_results_fpath = Path(save_dir) / "results.csv"

    with open(test_results_fpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())

        if not test_results_fpath.exists():
            writer.writeheader()
        writer.writerow(results_dict)


def main(cfg, overwrite=False):

    random_seed = cfg.experiment.random_seed
    checkpoint_path = cfg.model.checkpoint
    plot_examples_flag = cfg.experiment.plot_examples

    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set the seed for reproducibility
    set_seed(random_seed)

    save_dir = setup_save_dir(cfg, overwrite=overwrite, mode="eval")

    dataloaders = setup_data(cfg)
    test_loader = dataloaders['test']

    model = setup_model(cfg, device=device, mode='eval')
    load_checkpoint(checkpoint_path, model)

    loss_function = setup_loss_function(cfg)
    
    # Test model
    results = test(
        model, test_loader, loss_function, plot_examples=plot_examples_flag, save_dir=save_dir
    )
    logging.info(f"Test Loss: {results['loss']:.6f}")
    save_results(save_dir, results)

    return results


def test(model, test_loader, loss_function, plot_examples=False, eval_dir=None, n_sources=8):
    """
    Perform inference on a test set using the trained model.
    """
    totals = dict(loss=0.0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, roa=0.0)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
        
            inputs = inputs.to(device)
            targets = targets.to(device)
          
            output = model(inputs)
            loss = loss_function(output, targets)

            accuracy, precision, recall, f1, roa = metrics.get_metrics(output, targets) 
            

            totals['loss'] += loss.item()
            totals['accuracy'] += accuracy
            totals['precision'] += precision
            totals['recall'] += recall
            totals['f1'] += f1
            totals['roa'] += roa

            if plot_examples:
                vis.plot_eval(inputs, targets, output, fpath=eval_dir/f'test_{sample_num}.png')
                sample_num += 1
    
    num_batches = len(test_loader)
    results = {k: v / num_batches for k, v in totals.items()}

    logging.info(
        "\nTest set: " +
        " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in results.items()])
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    setup_logging()

    main(cfg, overwrite=args.overwrite)
