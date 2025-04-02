"""
Module main

Entry point for development ONLY

TODO:
    - Move to scripts/train.py scripts/evaluate.py 
    - Not sure if the loss functions work wrt format of the inputs (i.e. correct tensors)
    - Make config file robust to missing parameters
    - Add logging before raising exceptions
    - Check that random seed is set correctly. Does it need to be set in data_generator too?
    - Implement some more interpretable metric, such as accuracy within some tolerance
    - Fix model saving results with too many decimal places
    - Implement model saving and loading
    - Refactor

Notes:
    - reduction parameter cant be changed once the loss function is defined. 
        reduction mode is fixed when you create the loss function instance (loss_function = F.mse_loss).
    - reduction parameter has different effects for different loss functions. 
        reduction=mean will take average over ALL dimensions (not just batch size)
        It doesn't really matter how many dimensions it averages over, as it will just scale all losses.
        

Authors:

Usage:
    $ python -m src.main --config configs/config.yaml

Date Created: 
    19/12/2024
"""

from src.data.dataset import SyntheticDataset
from src.utils.visualisation import visualise_EMG, visualise_spike_trains
from src.models.black_box import DeconvolutionModel
from src.models.losses import Losses

from src.utils.utils import create_experiment_directory, change_log_file, plot_and_save_loss_curves

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm 
import yaml
import argparse
import os
import logging
import csv



torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def load_config(fname): 
    """
    Load a YAML conifguration file
    """
    
    with open(fname, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_data(config):
    """
    Create DataLoaders for training, validation and testing from config 
    """

    # Data generation config parameters
    total_samples = config['data']['total_samples']
    num_sources = config['data']['num_sources']
    num_channels = config['data']['num_channels']
    duration = config['data']['duration']
    sampling_frequency = config['data']['sampling_frequency']
    filter_type = config['data']['filter_type']

    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    # Get sizes of dataset split from config parameters
    train_size = int( train_split * total_samples)
    val_size = int(val_split * total_samples)
    test_size = total_samples - train_size - val_size

    # Create synthetic dataset from config file
    dataset = SyntheticDataset(
        num_samples=total_samples,
        sources=num_sources,
        channels=num_channels,
        duration=duration,
        sampling_frequency=sampling_frequency,
        filter_type=filter_type,
    )

    # Split data set into train test and validation
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['validation']['batch_size'], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['testing']['batch_size'], 
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def setup_model(config):
    """
    Create model from config file
    """

    num_sources = config['data']['num_sources']
    num_channels = config['data']['num_channels']
    duration = config['data']['duration']
    sampling_frequency = config['data']['sampling_frequency']
    time_steps = int(np.ceil(duration * sampling_frequency))

    if config['model']['name'].lower() == 'deconvolution':
        model = DeconvolutionModel(
            channels=num_channels,
            sources=num_sources,
            time_steps=time_steps,
            n_hidden=config['model']['n_hidden'],
        ).to(device)
    else:
        raise ValueError("Model not recognised")
    
    return model

def setup_optimiser(model, config):
    """
    Create optimiser from config file
    """

    learning_rate = config['training']['learning_rate']
    momentum = config['training']['momentum']

    if config['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum
        )
    elif config['training']['optimizer'].lower() == 'adam':
        pass
    else:
        raise ValueError("Optimizer not recognised")

    return optimizer

def setup_loss_function(config):
    """
    Create loss function from config file
    """
    losses = Losses()

    if config['training']['loss_function'].lower() == 'cross_entropy':
        loss_function = F.cross_entropy
    elif config['training']['loss_function'].lower() == 'mse':
        loss_function = F.mse_loss
    elif config['training']['loss_function'].lower() == 'total':
        # this includes positional, sparsity, num spikes and distance penalty terms 
        loss_function = losses.total_loss
    else:
        raise ValueError("Loss function not recognised")
    
    # Use Hungarian matching for loss function
    if config['training']['hungarian_matching']:
        loss_function = losses.hungarian_loss_wrapper(loss_fn=loss_function)

    return loss_function


def main():

    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")

    # Parse the arguments
    args = parser.parse_args()

    # Load the configuration from the YAML file
    config = load_config(args.config)

    # Set the seed for reproducibility
    seed = config['experiment']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup model and results saving
    if config['experiment']['save']:
        # Create experiment directory
        experiment_dir = create_experiment_directory(config)

        # Save logger to file
        change_log_file(experiment_dir, "train.log")

        # Save a copy of the config file
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

        # Create file to save training results
        train_results_fpath = os.path.join(experiment_dir, "results", "train_results.csv")
        with open(train_results_fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    train_loader, val_loader, test_loader = setup_data(config)

    model = setup_model(config)

    optimizer = setup_optimiser(model, config)

    loss_function = setup_loss_function(config)

    # Training results for saving and plotting
    # Add more metrics
    results = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
    }

    # Train-test loop
    for epoch in range(config['training']['num_epochs']):
        train_loss = train(model, 
              train_loader, 
              optimizer, 
              loss_function, 
              epoch, 
              log_interval=config['logging']['log_interval']
        )
        
        val_loss = test(
            model, 
            val_loader, 
            loss_function
        )

        # For plotting
        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss) 

        # Save results to file
        if config['experiment']['save']:
            with open(train_results_fpath, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss])

    # Save training loss plots to file
    if config['experiment']['save']:
        plot_and_save_loss_curves(results, experiment_dir)

    return


def train(model, train_loader, optimizer, loss_function, epoch, log_interval=100):
    """
    A utility function that performs a basic training loop.

    For each batch in the training set, fetched using `train_loader`:
        - Zeroes the gradient used by `optimizer`
        - Performs forward pass through `model` on the given batch
        - Computes loss on batch
        - Performs backward pass
        - `optimizer` updates model parameters using computed gradient

    Prints the training loss on the current batch every `log_interval` batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        # Send batch to the device we are using
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zeroes the gradient used by `optimizer`
        optimizer.zero_grad()

        # Performs forward pass through `model` on the given batch; equivalent
        # to `model.forward(inputs)`
        outputs = model(inputs)

        # Computes loss on batch
        # Use mean reduction to compute the average loss over the batch.
        # NOTE reduction parameter has different effects for different loss functions
        #       in terms of the number of dimensions it averages over.
        #       Doesnt really matter as long as left consistent for training (loss scaled by constant factor)
        loss = loss_function(outputs, targets, reduction="mean")

        # Performs backward pass; steps backward through the computation graph,
        # computing the gradient of the loss wrt model parameters.
        loss.backward()

        # `optimizer` updates model parameters using computed gradient.
        optimizer.step()

        total_loss += loss.item()

        # Prints the training loss on the current batch every `log_interval`
        # batches.
        if batch_idx % log_interval == 0:
            logging.info(
                "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                    epoch,
                    batch_idx,
                    # Calling `loss.item()` returns the scalar loss as a Python
                    # number.
                    loss.item(),
                )
            )
    
    # Note that the average loss is computed differently to the test funciton, 
    # but they both compute the same thing: the avereage loss over all datapoints in the dataset
    return total_loss / num_batches

def test(model, test_loader, loss_function):
    """
    A utility function to compute the loss and accuracy on a test set by
    iterating through the test set using the provided `test_loader` and
    accumulating the loss and accuracy on each batch.       
    """
    test_loss = 0.0
    test_loss_1 = 0.0
    correct = 0
    num_batches = len(test_loader)

    # You should use the `torch.no_grad()` context when you want to perform a
    # forward pass but do not need gradients. This effectively disables
    # autograd and results in fewer resources being used to perform the forward
    # pass (since information needed to compute gradients is not logged).
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            # We use `reduction="sum"` to aggregate losses across batches using
            # summation instead of taking the mean - we will take the mean at
            # the end once we have accumulated all the losses.
            # test_loss += loss_function(outputs, targets, reduction="sum").item()
            test_loss += loss_function(outputs, targets, reduction="mean").item()

            pred = (outputs >= 0.5).float()
            correct += pred.eq(targets).sum().item()

    # Divide by total number of samples to get average loss
    # test_loss /= len(test_loader.dataset) # * targets.shape[1] * targets.shape[2]
    test_loss /= num_batches

    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(
            test_loss, correct / len(test_loader.dataset)
        )
    )
    return test_loss


if __name__ == "__main__":
    main()