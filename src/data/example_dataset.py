from omegaconf import OmegaConf

from src.data import BaseDataset

import torch

class ExampleDataset(BaseDataset):
    """
    Example dataset class inheriting from BaseDataset.
    This is a placeholder for an actual dataset implementation.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        # Initialize dataset-specific attributes here

    def __len__(self):
        # Return the size of the dataset
        return 10000  # Example size

    def __getitem__(self, idx):
        # return random data example
        x = torch.randint(0, 256, (32,), dtype=torch.float32)
        y = torch.sum(x).fmod(2).unsqueeze(0).float()  # Example target
        return x, y