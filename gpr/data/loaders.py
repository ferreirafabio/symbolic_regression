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
from gpr.data.utils import all_tokens, token_to_index
from gpr.utils.configuration import Config
from gpr.data.data_creator import get_base_name
from sympy.parsing.latex import parse_latex


# TODO multi gpu, multi node loading
# TODO samller realization as in data arraw file
# TODO speed test


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
    def __init__(self, config_path, logger):
        global_config = Config(config_file=config_path)
        self.config = global_config.dataloader
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed # base seed for training set
        self.val_seed = self.seed + 1000  # base seed for validation set

        if logger is None:
            self.logger = logging.getLogger('logger')
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.ignore_index = -100
        self.pad_index = token_to_index['<PAD>']
        self.soe_index = token_to_index['<SOE>']
        self.eoe_index = token_to_index['<EOE>']
        self.token_to_index = token_to_index
        self.all_tokens = all_tokens

    def get_vocab(self):
        return self.all_tokens


    def indices_to_string(self, indices_batch):
        # if isinstance(indices, torch.Tensor):
        #     indices = indices.tolist()
        # return ''.join([all_tokens[i] for i in indices])
        """
        Convert a batch of sequences of indices into their corresponding LaTeX strings,
        respecting the actual length of each sequence.

        Args:
            indices_batch (torch.Tensor): A 2D tensor where each row is a sequence of indices.
            lengths (torch.Tensor): A tensor containing the lengths of each sequence in the batch.

        Returns:
            List[str]: A list of LaTeX strings corresponding to the sequences, truncated to their respective lengths.
        """
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.tolist()

        # Convert each sequence of indices into a LaTeX string, truncating by the corresponding length

        string_list = []
        for indices in zip(indices_batch):
            indices = indices[0]
            if self.eoe_index in indices:
                eoe_index = indices.index(self.eoe_index)
                indices = indices[:eoe_index]
                string = ''.join([self.all_tokens[i] for i in indices])
                string_list.append(string)
            else:
                string_list.append(None)

        return string_list
        # return [''.join([all_tokens[i] for i in indices[:length]]) for indices, length in zip(indices_batch, lengths)]

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
            self.logger.debug(f"Failed to parse LaTeX equation: {latex_equation}. Error: {e}")
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

    def compute_mse(self, predicted_seqs, true_seqs):
        """
        Compute the mean squared error (MSE) between batches of predicted and true LaTeX equations,
        only considering valid equations.

        Args:
            predicted_seqs (List[str]): A list of predicted LaTeX equation strings.
            true_seqs (List[str]): A list of true LaTeX equation strings.

        Returns:
            Tuple[float, int]: The average MSE over valid equations in the batch and the count of valid equations.
        """
        total_mse = 0.0
        valid_eq_count = 0

        for pred_seq, true_seq in zip(predicted_seqs, true_seqs):

            if pred_seq is None:
                self.logger.debug(f"End of equation token not found in predicted sequence, skipping MSE computation for this pair.")
                continue

            pred_eq, pred_vars = self.latex_equation_to_function(pred_seq)
            true_eq, true_vars = self.latex_equation_to_function(true_seq)

            # Check if equations are valid
            if pred_eq is None or true_eq is None:
                self.logger.debug(f"Invalid equations, skipping MSE computation for this pair.")
                continue

            self.logger.debug(f"pred_vars: {pred_vars} from pred_eq: {pred_eq} (latex string: {pred_seq})")
            self.logger.debug(f"true_vars: {true_vars} from true_eq: {true_eq} (latex string: {true_seq})")

            # Generate a union of all variables
            all_vars = list(set(pred_vars) | set(true_vars))

            # Generate values for variables
            var_values = {var: np.random.uniform(-10, 10, 10) for var in all_vars}

            try:
                pred_input_values = [var_values[var] for var in pred_vars]
                true_input_values = [var_values[var] for var in true_vars]

                # Log variable values
                # for var in pred_vars:
                #     self.logger.debug(f"Values for {var} in pred_eq: {var_values[var]}")
                # for var in true_vars:
                #     self.logger.debug(f"Values for {var} in true_eq: {var_values[var]}")

                # Create lambdified functions and evaluate them
                pred_func = sp.lambdify([var for var in pred_vars], pred_eq.rhs, modules=['numpy'])
                true_func = sp.lambdify([var for var in true_vars], true_eq.rhs, modules=['numpy'])

                pred_values = pred_func(*pred_input_values)
                true_values = true_func(*true_input_values)

                mse = np.mean((pred_values - true_values) ** 2)
                norm_true_values = np.linalg.norm(true_values)
                normalized_mse = mse / (norm_true_values + 1e-8)
                self.logger.debug(f"pred_values: {pred_values}, true_values: {true_values}, normalized MSE: {normalized_mse}")

                total_mse += normalized_mse
                valid_eq_count += 1

            except Exception as e:
                self.logger.debug(f"Failed to evaluate equations: {pred_seq} and {true_seq}")
                self.logger.debug(f"Error: {str(e)}")
                continue

        if valid_eq_count == 0:
            return 0.0, 0  # No valid equations, return 0 MSE and 0 valid equations

        return total_mse / valid_eq_count, valid_eq_count


    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch


    def index_pa_collator(self, indices, dataset, num_realizations):
        """get a set of samples. return a batch for tbl and trg_tex. pad target
        sequence with ignore index and input table with pad index."""

        batch = defaultdict(list)

        for i in indices:
            sample = dataset.get_record_batch(i)
            for key in ['mantissa', 'exponent', 'token_tensor']:
                array_data = np.array(sample[key].to_pylist()[0])
                tensor_data = torch.from_numpy(array_data)
                # Ensure mantissa and exponent are float32
                if key in ['mantissa', 'exponent']:
                    tensor_data = tensor_data[:num_realizations, :]
                    tensor_data = tensor_data.float()

                batch[key].append(tensor_data.t())  # Transpose for PyTorch

            batch['latex_expression'].append(sample['latex_expression'].to_pylist()[0])

        mantissa_stack = pad_sequence(batch['mantissa'],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1)
        exponent_stack = pad_sequence(batch['exponent'],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1).to(mantissa_stack.dtype)

        input_seq = [torch.cat([torch.tensor([self.soe_index], dtype=torch.long), tokens]) for tokens in batch['token_tensor']]
        input_seq_stack = pad_sequence(input_seq, batch_first=True, padding_value=self.pad_index)

        target_seq = [torch.cat([tokens, torch.tensor([self.eoe_index], dtype=torch.long)]) for tokens in batch['token_tensor']]
        target_seq_stack = pad_sequence(target_seq, batch_first=True, padding_value=self.ignore_index)

        return {"mantissa": mantissa_stack,
                "exponent": exponent_stack,
                "in_equation": input_seq_stack,
                "trg_equation": target_seq_stack,
                "latex_equation": batch['latex_expression'],
                'trg_len': torch.tensor([seq.shape[0] + 1 for seq in batch['token_tensor']])}


    def get_data_loader(self, set_name: str):
        """return a dataloader over an finite set of training data."""

        assert set_name in ['train', 'valid']

        base_name = get_base_name(self.config, set_name)

        file_name = f"{set_name}_{base_name}"
        data_dir = pathlib.Path(self.config.data_dir)
        file_dir = (data_dir / file_name).as_posix()

        if not os.path.exists(file_dir):
            available_files = os.listdir(data_dir)
            available_files_str = "\n".join(available_files) if available_files else "No files found."

            raise FileNotFoundError(
                f"Data file '{file_dir}' not found. Run the data creation script first.\n"
                f"Available files in '{data_dir}':\n{available_files_str}"
            )

        mmap = pa.memory_map(file_dir)
        self.logger.info("MMAP Read ALL")
        dataset = pa.ipc.open_file(mmap)

        train_set_size = dataset.num_record_batches

        indices = list(range(train_set_size))
        indices = np.random.permutation(indices)
        index_dataset = IndexDataset(indices)

        train_pl_collate_fn = partial(self.index_pa_collator, dataset=dataset, num_realizations=self.config.generator.num_realizations)

        data_loader = DataLoader(
            index_dataset,
            batch_size=self.config.batch_size,
            collate_fn=train_pl_collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True if set_name == 'train' else False,
        )
        
        self.logger.info(f"DataLoader for {set_name} successfully loaded from {file_dir}.")

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
        print(f"trg_len: {batch['trg_len']}")
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
