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
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng

from gpr.data.datasets import EquationDataset
from gpr.data.utils import all_tokens
from gpr.utils.configuration import Config
from sympy.parsing.latex import parse_latex


def clean_variable_names(equation):
    """Replace subscripted variable names with non-subscripted names."""
    lhs, rhs = equation.lhs, equation.rhs
    var_map = {}
    counter = 1

    # Sort variables to ensure consistent ordering
    sorted_vars = sorted(rhs.free_symbols, key=lambda v: str(v))

    for var in sorted_vars:
        var_str = str(var)
        if var_str == 'x':
            # if x and x_1 are present in predicted eq., do not use x_1 for x
            # since true eq. can have an x_1 at the same position of x_1 in predicted eq.
            var_map[var] = sp.Symbol('x')
        else:
            # For any other variables, including x_{i}, remove '_' and use x{counter}
            new_var = sp.Symbol(f'x{counter}')
            var_map[var] = new_var
            counter += 1

    new_rhs = rhs.subs(var_map)
    return sp.Eq(lhs, new_rhs), var_map



class IndexDataset(torch.utils.data.Dataset):
    """
    Wrapper class to hold arrow file dataset indices
    """

    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)


class SymPySimpleDataModule(object):
    def __init__(self, config_path, exp_folder):
        global_config = Config(config_file=config_path)
        self.config = global_config.dataloader
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed # base seed for training set
        self.val_seed = self.seed + 1000  # base seed for validation set

        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        exp_folder = pathlib.Path(exp_folder)
        os.makedirs(exp_folder, exist_ok=True)
        fh = logging.FileHandler(exp_folder / 'equation_log.txt')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Prevent the logger from propagating to the root logger
        self.logger.propagate = False

        self.ignore_index = -100
        self.pad_index = 0

    def get_vocab(self):
        return all_tokens


    def indices_to_string(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([all_tokens[i] for i in indices])

    @property
    def vocab_size(self):
        return len(self.get_vocab())

    def latex_equation_to_function(self, latex_equation):
        """Convert a LaTeX equation string to a SymPy function."""
        try:
            parsed_eq = self.parse_latex_equation(latex_equation)

            # clean variable names from underscored variables
            cleaned_eq, var_map = clean_variable_names(parsed_eq)
            variables = [var_map.get(var, var) for var in cleaned_eq.free_symbols if str(var) != 'y']

            return cleaned_eq, variables

        except Exception as e:
            self.logger.info(f"Failed to parse LaTeX equation: {latex_equation}. Error: {e}")
            return None, None

    def parse_latex_equation(self, latex_equation):
        """Check if a LaTeX equation string is valid by attempting to parse it."""

        # Quick check for balanced braces
        if latex_equation.count('{') != latex_equation.count('}'):
            raise ValueError("Unbalanced braces in LaTeX equation")

        try:
            parsed_eq = parse_latex(latex_equation)
            return parsed_eq

        except Exception as e:
            raise e

    def compute_mse(self, predicted_seq, true_seq):
        """Compute the mean squared error between two latex equations."""
        pred_eq, pred_vars = self.latex_equation_to_function(predicted_seq)
        true_eq, true_vars = self.latex_equation_to_function(true_seq)

        # check if equations are valid
        if pred_eq is None or true_eq is None:
            return float('inf')

        self.logger.info(f"pred_vars: {pred_vars} from pred_eq: {pred_eq} (latex string: {predicted_seq})")
        self.logger.info(f"true_vars: {true_vars} from true_eq: {true_eq} (latex string: {true_seq})")

        # Generate a union of all variables
        all_vars = list(set(pred_vars) | set(true_vars))

        # Generate values for variables
        # var_values = {(var): np.linspace(-10, 10, 50) for var in all_vars}
        var_values = {var: np.random.uniform(-10, 10, 10) for var in all_vars}

        try:
            pred_input_values = [var_values[(var)] for var in pred_vars]
            true_input_values = [var_values[(var)] for var in true_vars]

            # Log variable values
            for var in pred_vars:
                self.logger.info(f"Values for {var} in pred_eq: {var_values[var]}")
            for var in true_vars:
                self.logger.info(f"Values for {var} in true_eq: {var_values[var]}")

             # Create lambdified functions and evaluate them
            pred_func = sp.lambdify([var for var in pred_vars], pred_eq.rhs, modules=['numpy'])
            true_func = sp.lambdify([var for var in true_vars], true_eq.rhs, modules=['numpy'])

            pred_values = pred_func(*pred_input_values)
            true_values = true_func(*true_input_values)

            mse = np.mean((pred_values - true_values) ** 2)
            self.logger.info(f"pred_values: {pred_values}, true_values: {true_values}, MSE: {mse}")
            return mse
        except Exception as e:
            self.logger.info(f"Failed to evaluate equations: {predicted_seq} and {true_seq}")
            self.logger.info(f"Error: {str(e)}")
            return float('inf')

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch


    def index_pa_collator(self, indices, dataset):
        """get a set of samples. return a batch for tbl and trg_tex. pad target
        sequence with ignore index and input table with pad index."""

        batch = defaultdict(list)

        for i in indices:
            sample = dataset.get_record_batch(i)
            for key in ['mantissa', 'exponent', 'token_tensor']:
                batch[key].append(torch.from_numpy(np.array(sample[key].to_pylist()[0])).t())
            batch['latex_expression'].append(sample['latex_expression'].to_pylist()[0])

        mantissa_stack = pad_sequence(batch['mantissa'],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1)
        exponent_stack = pad_sequence(batch['exponent'],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1).to(mantissa_stack.dtype)
        latex_token_stack = pad_sequence(batch['token_tensor'],
                                         batch_first=True,
                                         padding_value=self.pad_index)
        return {"mantissa": mantissa_stack,
                "exponent": exponent_stack,
                "latex_token": latex_token_stack,
                "equation": batch['latex_expression']}

    def get_base_name(self):
        params = [
            f"s{self.config.generator.seed}",
            f"n{self.config.generator.num_nodes}",
            f"e{self.config.generator.num_edges}",
            f"t{self.config.generator.max_terms}",
            f"r{self.config.generator.num_realizations}",
            "real" if self.config.generator.real_numbers_realizations else "int"
        ]
        # Add allowed operations if present
        if self.config.generator.allowed_operations:
            char_map = {'+': 'plus', '-': 'minus', '*': 'mul', '/': 'div'}
            # Replace unsafe characters and join operations
            safe_ops = [char_map.get(op, op) for op in self.config.generator.allowed_operations]
            ops = "_".join(sorted(safe_ops))
            params.append(f"ops_{ops}")

        # Join all parameters
        base_name = f"{'_'.join(params)}.arrow"
        return base_name

    def get_data_loader(self, set_name: str):
        """return a dataloader over an infinite set of training data."""

        assert set_name in ['train', 'valid']

        base_name = self.get_base_name()

        file_name = f"{set_name}_{base_name}"
        data_dir = pathlib.Path(self.config.data_dir)
        file_dir = (data_dir / file_name).as_posix()

        if not os.path.exists(file_dir):
            raise FileNotFoundError(f"Data file {file_dir} not found. Run data creation script first.")

        mmap = pa.memory_map(file_dir)
        self.logger.info("MMAP Read ALL")
        dataset = pa.ipc.open_file(mmap)

        train_set_size = dataset.num_record_batches

        num_cpu_worker = 4
        batch_size = 4

        indexes = list(range(train_set_size))
        indexes = np.random.permutation(indexes)
        index_dataset = IndexDataset(indexes)

        train_pl_collate_fn = partial(self.index_pa_collator, dataset=dataset)

        data_loader = DataLoader(
            index_dataset,
            batch_size=batch_size,
            collate_fn=train_pl_collate_fn,
            num_workers=num_cpu_worker,
            pin_memory=True,
            drop_last=True if set_name == 'train' else False,
        )

        return data_loader




    def get_train_loader(self):
        """return a dataloader over an infinite set of training data."""

        train_loader = self.get_data_loader(set_name='train')
        return train_loader

    def get_valid_loader(self):

        valid_loader = self.get_data_loader(set_name='valid')
        return valid_loader


if __name__ == "__main__":

    sympy_data = SymPySimpleDataModule(config_path='config/default_config.yaml', exp_folder='exp')
    train_loader = sympy_data.get_train_loader()
    valid_loader = sympy_data.get_valid_loader()

    print("Validation equations:")
    counter = 0
    for batch in valid_loader:
        print(f"Batch {counter} equations:")
        print(f"mantissa: {batch['mantissa'].shape}")
        print(f"exponent: {batch['exponent'].shape}")
        print(f"latex_token: {batch['latex_token'].shape}")
        for equation in batch['equation']:
            print(equation)
        counter += 1
        if counter == 5:
            break

    print("Training equations:")
    counter = 0
    for batch in train_loader:
        print(f"Batch {counter} equations:")
        for equation in batch['equation']:
            print(equation)
        counter += 1
        if counter == 5:
            break
