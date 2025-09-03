""" Module setup

Provides functions to set up the dataset and dataloaders automatically

Usage:
    from src.data.setup import setup_data
    from src.data import setup_data

    dataloaders, scheduler = setup_data(cfg, mode="train")

Authors:
    Will Raftery

Date:
    31/07/2025
"""

import importlib
from omegaconf import OmegaConf
from src.data.curriculum_scheduler import CurriculumScheduler

from torch.utils.data import DataLoader, random_split

import logging


def find_dataset_using_name(dataset_name: str):
    """
    Import the module "[dataset_name]_dataset.py".
    The class [DatasetName]Dataset must exist inside (future: and inherit from BaseDataset).
    """
    module_name = f"src.data.{dataset_name}_dataset"
    try:
        datasetlib = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module {module_name}") from e

    target_class_name = f"{dataset_name.capitalize()}Dataset"

    for name in dir(datasetlib):
        obj = getattr(datasetlib, name)
        if name.lower() == target_class_name.lower() and isinstance(obj, type): # issubclass(obj, BaseDataset):
            return obj

    raise ImportError(f"Expected class '{target_class_name}' in '{module_name}'.")


def setup_data(cfg: OmegaConf, mode: str = "train"):
    """
    Setup data loaders and scheduler based on the configuration.
    
    Args:
        cfg (OmegaConf): Configuration object
        mode (str): One of ["train", "val", "test", "predict"]

    Returns:
        dataloaders (dict): Dict of DataLoaders (train/val/test) as appropriate
        scheduler (optional): Curriculum scheduler for training (if applicable)
    """
    num_workers = cfg.data.num_workers
    dataset_name = cfg.data.name

    dataloaders = {}
    scheduler = None

    # Dynamically load dataset class
    dataset_class = find_dataset_using_name(dataset_name)

    if mode == "train":
        val_split = cfg.training.val_split
        train_batch_size = cfg.training.batch_size
        val_batch_size = cfg.validation.batch_size

        dataset = dataset_class(cfg)

        train_dataset, val_dataset = random_split(dataset, [(1 - val_split), val_split])
            
        # Create DataLoaders for training and validation
        dataloaders['train'] = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
        dataloaders['val'] = DataLoader(
            val_dataset, 
            batch_size=val_batch_size, 
            shuffle=True,
            num_workers=num_workers
        )

        if cfg.data.curriculum.flag:
            scheduler = CurriculumScheduler(
                dataset=dataset,
                milestones=cfg.data.curriculum.milestones,
                parameters=cfg.data.curriculum.parameters
            )
        else:
            scheduler = None

        logging.info(f"Data setup complete: {len(dataloaders['train'])} training batches, {len(dataloaders['val'])} validation batches.")

        return dataloaders, scheduler
    
    elif mode == "test":

        # Placeholder for actual data setup logic
        # This should return train_loader, val_loader, scheduler
        raise NotImplementedError("Data setup function is not implemented.")

    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are ['train', 'test'].")