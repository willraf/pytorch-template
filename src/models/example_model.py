""" Module example_dataset

Example dataset module for demonstration purposes.

Usage:


Authors:
    Will Raftery

Date:
    31/07/2025
"""

from omegaconf import OmegaConf
import torch.nn as nn

class ExampleModel(nn.Module):

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )


    def forward(self, x):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        return self.layers(x)