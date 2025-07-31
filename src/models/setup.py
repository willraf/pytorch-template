"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel
"""

import importlib
from torch.optim import lr_scheduler
from src.utils.config_wrapper import Config
import logging


def find_model_using_name(model_name: str):
    """
    Import the module "models/[model_name]_model.py".
    The class [ModelName]Model must exist inside and inherit from BaseModel.
    """
    module_name = f"models.{model_name}_model"
    try:
        modellib = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module {module_name}") from e

    target_class_name = f"{model_name.capitalize()}Model"

    for name in dir(modellib):
        obj = getattr(modellib, name)
        if name.lower() == target_class_name.lower() and isinstance(obj, type): # issubclass(obj, BaseDataset):
            return obj

    raise ImportError(f"Expected class '{target_class_name}' in '{module_name}'.")


def setup_model(cfg: Config):
    """Create a model given the configuration.

    This is the main interface between this package and train.py/validate.py
    """
    model_name = cfg.get("model/name", required=True)
    model = find_model_using_name(model_name)
    instance = model(cfg)
    logging.info("model [{0}] was created".format(type(instance).__name__))
    return instance