from omegaconf import OmegaConf

from src.data import BaseDataset

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
        return 1000  # Example size

    def __getitem__(self, idx):
        # Return a single data item
        return {"data": idx, "label": idx % 10}  # Example data and label