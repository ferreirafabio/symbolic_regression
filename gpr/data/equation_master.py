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
import re
import yaml
import random
import logging
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng

from gpr.data.utils import format_floats_recursive
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


class EquationMaster(object):
    def __init__(self, global_config, logger):
        self.config = global_config.dataloader
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed  # base seed for training set
        # self.val_seed = self.seed + 1000  # base seed for validation set

        self.ignore_index = -100
        self.pad_index = token_to_index['<PAD>']
        self.soe_index = token_to_index['<SOE>']
        self.eoe_index = token_to_index['<EOE>']
        self.token_to_index = token_to_index
        self.all_tokens = all_tokens

        self.real_const_decimal_places = self.config.generator.real_const_decimal_places

        if logger is None:
            self.logger = logging.getLogger('logger')
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        # Get the directory from the existing log file
        log_file_name = self._get_log_file_name()
        log_directory = pathlib.Path(log_file_name).parent

        # Set up the equation logger in the same directory
        equation_log_file = log_directory / 'equation_log.txt'
        self.equation_logger = self._setup_equation_logger(equation_log_file)

    def _get_log_file_name(self):
        # Loop through all handlers and check if they are FileHandlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                # Return the file name from the handler
                return handler.baseFilename
        return None

    def _setup_equation_logger(self, log_file_path):
        # Create a separate logger for equations
        equation_logger = logging.getLogger('equation_logger')
        equation_logger.setLevel(logging.INFO)

        # Create file handler for equation logging
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handler to the equation logger
        equation_logger.addHandler(fh)
        # Prevent log messages from propagating to the root/general logger
        equation_logger.propagate = False

        return equation_logger


    def latex_equation_to_function(self, latex_equation):
        """Convert a LaTeX equation string to a SymPy function."""
        try:
            parsed_eq = self.parse_latex_equation(latex_equation)
            self.equation_logger.info(f"Parsed equation: {parsed_eq}")

            # clean variable names from underscored variables
            cleaned_eq, var_map = clean_variable_names(parsed_eq)
            self.equation_logger.info(f"Cleaned equation: {cleaned_eq}")

            # Round the coefficients in the cleaned equation to the desired precision
            cleaned_eq = format_floats_recursive(cleaned_eq, self.real_const_decimal_places)

            variables = [var_map.get(var, var) for var in cleaned_eq.free_symbols if str(var) != 'y']

            return cleaned_eq, variables

        except Exception as e:
            self.equation_logger.info(f"Failed to parse LaTeX equation: {latex_equation}. Error: {e}")
            return None, None

    def parse_latex_equation(self, latex_equation):
        """Check if a LaTeX equation string is valid by attempting to parse it."""

        # Quick check for balanced braces
        if latex_equation.count('{') != latex_equation.count('}'):
            raise ValueError("Unbalanced braces in LaTeX equation")

        # Add explicit multiplication symbol * before parentheses to avoid
        # sympy misinterpreting x_{2}(..) as a function call
        latex_equation = re.sub(r'(\w+)(\()', r'\1*\2', latex_equation)

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
                self.equation_logger.info(
                    f"End of equation token not found in predicted sequence, skipping MSE computation for this pair (true: {true_seq}, pred.: {pred_seq}).")
                continue

            self.equation_logger.info(f"Predicted sequence: {pred_seq}")
            self.equation_logger.info(f"True sequence: {true_seq}")

            pred_eq, pred_vars = self.latex_equation_to_function(pred_seq)
            true_eq, true_vars = self.latex_equation_to_function(true_seq)

            # Check if equations are valid
            if pred_eq is None or true_eq is None:
                self.equation_logger.info(
                    f"Invalid equations, skipping MSE computation for this pair (true: {true_seq}, pred.: {pred_seq}).")
                continue

            self.equation_logger.info(f"pred_vars: {pred_vars} from pred_eq: {pred_eq} (latex string: {pred_seq})")
            self.equation_logger.info(f"true_vars: {true_vars} from true_eq: {true_eq} (latex string: {true_seq})")

            # Generate values for variables
            var_values = {var: np.round(np.random.uniform(-10, 10, 10), self.real_const_decimal_places) for var in
                          set(pred_vars) | set(true_vars)}

            try:
                pred_input_values = [var_values[var] for var in pred_vars]
                true_input_values = [var_values[var] for var in true_vars]

                # Create custom lambdify functions with controlled precision
                def rounded_lambdify(expr, variables):
                    lambda_func = sp.lambdify(variables, expr, modules=['numpy'])

                    def wrapper(*args):
                        result = lambda_func(*args)
                        return np.round(result, self.real_const_decimal_places)

                    return wrapper

                pred_func = rounded_lambdify([var for var in pred_vars], pred_eq.rhs)
                true_func = rounded_lambdify([var for var in true_vars], true_eq.rhs)

                pred_values = pred_func(*pred_input_values)
                true_values = true_func(*true_input_values)

                mse = np.mean((pred_values - true_values) ** 2)
                norm_true_values = np.linalg.norm(true_values)
                normalized_mse = mse / (norm_true_values + 1e-8)
                self.equation_logger.info(
                    f"Computed MSE successfully for: pred_values: {pred_values} and true_values: {true_values} with normalized MSE: {normalized_mse}")

                total_mse += normalized_mse
                valid_eq_count += 1

            except SyntaxError as se:
                self.equation_logger.info(f"SyntaxError: {se} for true equation {true_eq} and pred. equation {pred_eq}")

            except Exception as e:
                self.equation_logger.info(f"Failed to evaluate equations: {pred_seq} and {true_seq}")
                self.equation_logger.info(f"Error: {str(e)}")
                continue

        if valid_eq_count == 0:
            return 0.0, 0  # No valid equations, return 0 MSE and 0 valid equations

        return total_mse / valid_eq_count, valid_eq_count









if __name__ == "__main__":

    sympy_data = EquationMaster(config_path='config/default_config.yaml', exp_folder='exp')
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
