""" Module dataset

Contains the SyntheticEMGDataset class,
a custom pytorch dataset for integration with DataLoader
PyTorch specific interface to synthetic data generation.

Usage:

Authors:

Date:
    26/11/2024
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.data.data_generator import DataGenerator

class SyntheticDataset(Dataset):
    """

    """
    def __init__(
            self, 
            num_samples,
            sources, 
            channels, 
            duration, 
            sampling_frequency,
            filter_type="g"
        ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
        """
        self.num_samples = num_samples
        self.gen = DataGenerator(sources=sources, 
                                channels=channels, 
                                duration=duration, 
                                sampling_frequency=sampling_frequency)

        self.sources = sources
        self.channels = channels


    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.num_samples


    def __getitem__(self, idx):
        """
        Generate synthetic data and corresponding label for a given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the synthetic data (input) and the label (target).
        """

        # Randomly generate parameters
        firing_rates = np.random.randint(5, 60, (self.gen.num_sources))
        L = 30

        # Constant filters for now
        # Shape (m, n)
        mu_H = np.random.uniform(0.005, 0.1, size=(self.gen.num_channels, self.gen.num_sources))
        sigma_H = np.random.uniform(0.008, 0.018, size=(self.gen.num_channels, self.gen.num_sources))

        if firing_rates.shape[0] != self.gen.num_sources:
            raise ValueError("Incorrect number of firing rates provided given sources")

        if mu_H.shape != sigma_H.shape:
            raise ValueError("Mean and std parameters of filters do not match shape")

        if mu_H.shape[0] != self.gen.num_channels:
            raise ValueError("Incorrect number of parameters for channels")

        if mu_H.shape[1] != self.gen.num_sources:
            raise ValueError("Incorrect number of parameters for sources")


        emg, spike_trains, _ = self.gen.generate_data(firing_rates=firing_rates, 
                                filter_length=L, 
                                mu_H=mu_H, 
                                sigma_H=sigma_H)


        # Ensure outputs are torch.float32
        emg = torch.tensor(emg, dtype=torch.float32)
        spike_trains = torch.tensor(spike_trains, dtype=torch.float32)

        return emg, spike_trains





