"""
Module black_box

Contains DeconvolutionModel class for a basic black box model

Also contains skeleton code to outline project structure

TODO:
    Refactor into different files

Authors:

Date created:
    16/12/2024
"""


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class DeconvolutionModel(nn.Module):
    """
    Define the deconvolution model
    """

    def __init__(self, channels, sources, time_steps, n_hidden):
        """ Constructor

        TODO:
            Pass model hyperparameters and define model layers, variables etc. 

        Args:
            channels (int): The number of input channels
            sources (int): Number of sources to output 
            time_steps (int): Number of input time steps
            n_hidden (int): Dimentionality of hidden layer
        """
        super().__init__()

        self.n_channels = channels
        self.n_sources = sources 
        self.time_steps = time_steps

        self.fc1 = nn.Linear(in_features=(channels * time_steps), out_features=n_hidden, bias=True)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=(sources * time_steps), bias=True)


    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.tensor): Shape (m, T). m channels, T time steps
        """
        h = x.view(-1, (self.n_channels * self.time_steps))
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.sigmoid(h)
        return h.view(x.size(0), self.n_sources, self.time_steps)


if __name__ == "__main__":

    single_source_channel()