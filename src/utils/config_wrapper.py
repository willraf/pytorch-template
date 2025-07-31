"""
Module config_wrapper

Contains a wrapper class for configuration files, allowing for easy retrieval of nested elements.
"""

import copy
import yaml


def merge_dictionaries_recursively(dict1, dict2):
    """ Update two config dictionaries recursively.

    Args:
    dict1 (dict): first dictionary to be updated
    dict2 (dict): second dictionary which entries should be preferred
    """
    if dict2 is None: return

    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_dictionaries_recursively(dict1[k], v)
        else:
            dict1[k] = v


class Config(object):  
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """
    def __init__(self, config_path, default_path=None):
        with open(config_path, "r", encoding="utf-8") as cf_file:
            cfg = yaml.safe_load(cf_file)

        if default_path is not None:
            with open(default_path, "r", encoding="utf-8") as def_cf_file:
                default_cfg = yaml.safe_load(def_cf_file)
                    
            merge_dictionaries_recursively(default_cfg, cfg)
        
        self._data = cfg

    def get(self, path=None, default=None, required=False):
        """ Retrieve a value from the config dictionary

        Args:
            path (str): slash-separated path to the value, e.g. "training/epochs"
            default: default value to return if the path is not found
            required (bool): if True, raises an error if the path is not found, or the value is None
        """
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = copy.deepcopy(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            if required and value is None:
                raise ValueError(f"Required config parameter '{path}' is missing or None.")

            return value
        except (TypeError, AttributeError):

            if required:
                raise ValueError(f"Required config path '{path}' is invalid or missing.")   
            
            return default
        
        
    def save(self, path):
        """ Save the config dictionary to a YAML file

        Args:
            path (str): path to the file where the config should be saved
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)



