from typing import List, Tuple
from collections import defaultdict
from functools import partial
import os
import pathlib
import pyarrow as pa
import multiprocessing as mp

import sympy as sp
import pandas as pd
import numpy as np
import torch
import yaml
import random
import logging

from torch.onnx.symbolic_opset11 import chunk
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng
from tqdm import tqdm

from gpr.data.generators import FeynmanGenerator
from gpr.data.data_creator import CreateDataset

from gpr.utils.configuration import Config
from sympy.parsing.latex import parse_latex
from gpr.data.utils import tokenize_latex_to_char

## TODO dedub
VERSION = 1


def get_base_name_test(config, dataset_type):
    # Determine file extension based on the dataset_type
    if dataset_type == "config":
        extension = ".yaml"
    else:
        extension = ".arrow"

    # Join all parameters and return the base name
    base_name = f"{dataset_type}_v{VERSION}{extension}"
    return base_name

class CreateFeynmanDataset(CreateDataset):
    def __init__(self, config_file=None, config_dict=None, force_creation=True):

        self.force_creation = force_creation

        if config_file is None and config_dict is None:
            raise UserWarning("ConfigHandler: config_file and config_dict is None")

        global_config = Config(config_file=config_file, config_dict=config_dict)
        self.config = global_config.dataloader

        self.seed = self.config.generator.seed  # base seed for training set
        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.logger = logging.getLogger(__name__)

        self.rng = default_rng(self.seed)
        self.generator = FeynmanGenerator(rng=self.rng)

        num_cpus = mp.cpu_count()
        workers = (num_cpus // 2) & ~1

        data_dir = pathlib.Path(self.config.data_dir)
        os.makedirs(data_dir, exist_ok=True)

        test_base_name = get_base_name_test(self.config, "feynman")
        test_file_name = f"test_{test_base_name}"
        test_file_dir = (data_dir / test_file_name).as_posix()
        self.create_set(test_file_dir, workers, self.config.test_samples,
                        force_creation, dataset_type="test")

        base_name = get_base_name_test(self.config, "dataset")
        self.save_config(data_dir, base_name)

        self.pad_index = 0




if __name__ == "__main__":
    """
    example usage:
    python gpr/data/test_data_creator.py.py -c data_config -f
    """
    import argparse
    import socket
    import yaml
    import collections

    from functools import reduce  # forward compatibility for Python 3
    import operator


    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)

    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

    def convert_string_value(value):
        if value in ("false", "False"):
            value = False
        elif value in ("true", "True"):
            value = True
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        return value


    default_config_name = "default_config.yaml"

    parser = argparse.ArgumentParser(description="Generate Dataset")
    parser.add_argument("-c", "--config", type=str, default=default_config_name, help="config file name")
    parser.add_argument("-f", "--force_creation", action="store_true", help="force creation of dataset")

    args, unknown_args = parser.parse_known_args()

    config_name = args.config
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_file = os.path.join("config", config_name)
    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    for arg in unknown_args:
        if "=" in arg:
            keys = arg.split("=")[0].split(".")
            value = convert_string_value(arg.split("=")[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")

    config = Config(config_dict=config_dict)

    sympy_data = CreateFeynmanDataset(config_dict=config_dict, force_creation=args.force_creation)



