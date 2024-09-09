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

from gpr.data.generators import PolynomialGenerator

from gpr.utils.configuration import Config
from sympy.parsing.latex import parse_latex
from gpr.data.utils import tokenize_latex_to_char

## TODO dedub
VERSION = 1

def get_base_name(config, dataset_type):
    """
    Generate a base name for the file based on the configuration and dataset type.
    
    Parameters:
        config: The configuration object.
        dataset_type: The type of dataset (e.g., 'train', 'valid', 'config').
        
    Returns:
        The base file name as a string.
    """
    params = [
        f"smpls{config.train_samples if dataset_type == 'train' else config.valid_samples}",
        f"s{config.generator.seed}",
        f"n{config.generator.num_variables}",
        f"t{config.generator.max_terms}",
        f"dp{config.generator.real_const_decimal_places}",
        f"mc{config.generator.use_math_constants}",
        f"me{config.generator.max_const_exponent}",
        f"mp{config.generator.max_powers}",
        f"kmax{config.generator.kmax}",
        f"rcmin{config.generator.real_constants_min}",
        f"rcmax{config.generator.real_constants_max}",
    ]
    
    # Add allowed operations if present
    if config.generator.allowed_operations:
        char_map = {'+': 'plus', '-': 'minus', '*': 'mul', '/': 'div'}
        safe_ops = [char_map.get(op, op) for op in config.generator.allowed_operations]
        ops = "_".join(sorted(safe_ops))
        params.append(f"ops_{ops}")
    
    # Determine file extension based on the dataset_type
    if dataset_type == "config":
        extension = ".yaml"
    else:
        extension = ".arrow"
    
    # Join all parameters and return the base name
    base_name = f"{'_'.join(params)}_v{VERSION}{extension}"
    return base_name




class CreateDataset(object):
    def __init__(self, config_file=None, config_dict=None, force_creation=True, create_valid=True):

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

        if self.config.generator_type == 'RandomGenerator':
            raise NotImplementedError
            #self.generator = RandomGenerator(rng=self.rng)
        elif self.config.generator_type == 'PolynomialGenerator':
            self.generator = PolynomialGenerator(rng=self.rng)

        # num_cpus = mp.cpu_count()
        # workers = (num_cpus // 2) & ~1
        workers = mp.cpu_count()
        print(f"Using {workers} workers")

        data_dir = pathlib.Path(self.config.data_dir)
        os.makedirs(data_dir, exist_ok=True)

        if "project_name" in self.config and self.config.project_name:
            data_dir = data_dir / self.config.project_name
            os.makedirs(data_dir, exist_ok=True)


        train_base_name = get_base_name(self.config, "train")
        train_file_name = f"train_{train_base_name}"
        train_file_dir = (data_dir / train_file_name).as_posix()
        self.create_set(train_file_dir, workers, self.config.train_samples, force_creation, dataset_type="train")

        if create_valid:
            valid_base_name = get_base_name(self.config, "valid")
            valid_file_name = f"valid_{valid_base_name}"
            valid_file_dir = (data_dir / valid_file_name).as_posix()
            self.create_set(valid_file_dir, workers, self.config.valid_samples, force_creation, dataset_type="valid")

        base_name = get_base_name(self.config, "dataset")
        self.save_config(data_dir, base_name)

        self.pad_index = 0

    def save_config(self, data_dir, base_name):
        """
        Save the configuration used to generate the dataset as a YAML file.
        """
        # Get the base name specifically for the config file
        base_name = get_base_name(self.config, dataset_type="config")
        
        config_name = base_name
        config_path = data_dir / config_name
        
        with open(config_path, 'w') as yaml_file:
            yaml.dump(self.config, yaml_file)
        
        print(f"Configuration saved to {config_path}")


    def _create_sample(self):
        mantissa, exponent, expression, is_nan = self.generator(**self.config.generator)
        if mantissa is None:
            return None
        mantissa, exponent = mantissa.to(torch.float16), exponent.to(torch.int8)
        latex_expression = sp.latex(expression)
        latex_token_indices = tokenize_latex_to_char(latex_expression)
        token_tensor = torch.tensor(latex_token_indices, dtype=torch.uint8)
        sample = [mantissa.numpy(), exponent.numpy(), token_tensor.numpy(), is_nan.numpy()]
        return sample


    def _samples2queue(self, mp_queue, num_samples):
        try:
            for _ in range(num_samples):
                sample = self._create_sample()
                mp_queue.put(sample)
        except StopIteration:
            pass
        # finally:
        #     mp_queue.put('END')
        mp_queue.put('END')



    @staticmethod
    def _queue2file_writer(file_dir, schema, mp_queue, workers, total_samples, dataset_type="train"):
        sample_counter = 0
        with pa.OSFile(file_dir, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                try:
                    end_counter = 0
                    with tqdm(total=total_samples, desc=f"Writing {dataset_type} dataset", unit="samples") as pbar:
                        while True:
                            sample = mp_queue.get(timeout=10)
                            if sample == 'END':
                                end_counter += 1
                                if end_counter == workers:
                                    break
                            elif sample is None:
                                continue
                            else:
                                mantissa_batch, exponent_batch, token_tensor_batch, is_nan_batch = sample

                                batch = pa.RecordBatch.from_arrays([
                                    pa.array([[list(row) for row in mantissa_batch]]),
                                    pa.array([[list(row) for row in exponent_batch]]),
                                    pa.array([token_tensor_batch]),
                                    pa.array([is_nan_batch]),
                                ], schema=schema)
                                writer.write_batch(batch)
                                sample_counter += 1
                                pbar.update(1)
                finally:
                    mp_queue.close()
        print(f"queue2file_writer wrote total {sample_counter} in {file_dir}")


    def create_set(self, file_dir, workers, num_samples, force_creation, dataset_type="train"):

        if not os.path.exists(file_dir) or force_creation:
            schema = pa.schema([
                ('mantissa', pa.list_(pa.list_(pa.float16()))),
                ('exponent', pa.list_(pa.list_(pa.int8()))),
                ('token_tensor', pa.list_(pa.uint8())),
                ('is_nan', pa.list_(pa.bool_())),
            ])

            mp_manager_list = []
            mp_queue = mp.Queue(maxsize=workers*2)

            self.logger.info(f"Starting to write the {dataset_type} dataset to {file_dir}")
            print(f"Starting the creation of {dataset_type} dataset named {file_dir}...")
            mp_manager = mp.Process(target=self._queue2file_writer, args=(file_dir, schema, mp_queue, workers, num_samples, dataset_type))
            mp_manager.daemon = True
            mp_manager.start()
            mp_manager_list.append(mp_manager)

            samples_per_worker = num_samples // workers
            remaining_samples = num_samples % workers

            for worker_idx in range(workers):
                worker_samples = samples_per_worker + (1 if worker_idx < remaining_samples else 0)
                self.logger.info(f"Starting create_sample process {worker_idx} for {dataset_type} dataset with {worker_samples} samples")
                mp_manager = mp.Process(target=self._samples2queue, args=(mp_queue, worker_samples))
                mp_manager.daemon = True
                mp_manager.start()
                mp_manager_list.append(mp_manager)

            for mp_manager in mp_manager_list:
                mp_manager.join()
        return True



if __name__ == "__main__":
    """
    example usage:
    python gpr/data/data_creator.py.py -c data_config -f
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
    parser.add_argument("-v", "--no_valid", action="store_false", help="no validation set")
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

    sympy_data = CreateDataset(config_dict=config_dict, force_creation=args.force_creation, create_valid=args.no_valid)



