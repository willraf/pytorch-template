"""
Script predict

Handles model inference.

Usage:
    $ python -m scripts.predict --config configs/predict.yaml
    $ python -m scripts.predict --config configs/predict.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import logging
import time
import argparse

from src.utils.utils import *
from src.utils.config_wrapper import Config
import src.utils.visualisation as vis
from src.data import setup_data
from src.models import setup_model

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main(cfg, overwrite=False):

    random_seed = cfg.get("experiment/random_seed", default=None, required=True)
    checkpoint_path = cfg.get("model/checkpoint", default=None, required=True)
    plot_examples_flag = cfg.get("experiment/plot_examples", default=True, required=False)

    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set the seed for reproducibility
    set_seed(config['experiment']['random_seed'])

    save_dir = setup_save_dir(config, overwrite=overwrite)

    model = setup_model(cfg, device=device, mode='inference')
    load_checkpoint(checkpoint_path, model)

    dataloader = setup_data(cfg, mode="inference")
    infer_loader = dataloader['inference']

    predictions = predict(model, infer_loader, plot_examples=plot_examples_flag, save_dir=save_dir)

    # Save predictions and compute any metrics if necessary


def predict(model, infer_loader, plot_examples=True, plot_num=10, save_dir=None):
    """
    Perform inference on a test set using the trained model.
    """
    start_time = time.time()
    num_batches = len(infer_loader)
    plot_interval = num_batches // plot_num

    predictions = []

    model.eval()
    with torch.no_grad():

        for batch_idx, inputs in enumerate(infer_loader):

            inputs = inputs.to(device)
            
            output = model(inputs)   

            if batch_idx % plot_interval == 0 and plot_examples:
                vis.plot_inference(inputs, output, fpath=save_dir / f'{batch_idx}.png')

            predictions.append(output.detach().cpu().numpy())

    total_time = time.time() - start_time

    logging.info(f'Inference completed in {total_time:.4f} seconds')
    avg_time = total_time / len(infer_loader)
    logging.info(f'Average time per window: {avg_time:.4f} seconds')

    return np.stack(predictions, axis=0)     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    config = Config(args.config, default_path=args.default)
    main(config, overwrite=args.overwrite)