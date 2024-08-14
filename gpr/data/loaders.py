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


class SymPySimpleDataModule(object):
    def __init__(self, generator, config_path, exp_folder):
        global_config = Config(config_file=config_path)
        self.config = global_config.dataloader
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed # base seed for training set
        self.val_seed = self.seed + 1000  # base seed for validation set
        self.generator_class = generator
        self.generators = [None] * self.config.num_workers

        self.equation_logger = logging.getLogger('equation_logger')
        self.equation_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(exp_folder / 'equation_log.txt')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.equation_logger.addHandler(fh)

        # Prevent the equation_logger from propagating to the root logger
        self.equation_logger.propagate = False

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
            self.equation_logger.info(f"Failed to parse LaTeX equation: {latex_equation}. Error: {e}")
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

        self.equation_logger.info(f"pred_vars: {pred_vars} from pred_eq: {pred_eq} (latex string: {predicted_seq})")
        self.equation_logger.info(f"true_vars: {true_vars} from true_eq: {true_eq} (latex string: {true_seq})")

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
                self.equation_logger.info(f"Values for {var} in pred_eq: {var_values[var]}")
            for var in true_vars:
                self.equation_logger.info(f"Values for {var} in true_eq: {var_values[var]}")

             # Create lambdified functions and evaluate them
            pred_func = sp.lambdify([var for var in pred_vars], pred_eq.rhs, modules=['numpy'])
            true_func = sp.lambdify([var for var in true_vars], true_eq.rhs, modules=['numpy'])

            pred_values = pred_func(*pred_input_values)
            true_values = true_func(*true_input_values)

            mse = np.mean((pred_values - true_values) ** 2)
            self.equation_logger.info(f"pred_values: {pred_values}, true_values: {true_values}, MSE: {mse}")
            return mse
        except Exception as e:
            self.equation_logger.info(f"Failed to evaluate equations: {predicted_seq} and {true_seq}")
            self.equation_logger.info(f"Error: {str(e)}")
            return float('inf')

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch

    def create_sample(self, is_validation=False):
        """Return a tbl of n inference samples of the equation, and the target
        token sequnce of the latex equation."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            if self.generators[worker_id] is None:
                raise RuntimeError(f"Generator for worker {worker_id} is not initialized.")
            generator = self.generators[worker_id]
        else:
            # num_workers <= 1 or valid dataloader case
            seed = self.val_seed if is_validation else self.seed
            rng = default_rng(seed)
            generator = self.generator_class(rng=rng)
        return generator(**self.config.generator)


    def create_validation_set(self):
        validation_data = [self.create_sample(is_=self.rng) for _ in range(self.config.val_samples)]
        validation_dataset = EquationDataset(
            data_source=validation_data,
            generator=self.generator
        )
        return validation_dataset

    def collator(self, batch):
        """get a set of samples. return a batch for tbl and trg_tex. pad target
        sequence with ignore index and input table with pad index."""
        mantissa_stack = pad_sequence([item[0].t() for item in batch],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1)
        exponent_stack = pad_sequence([item[1].t() for item in batch],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1).to(mantissa_stack.dtype)
        latex_token_stack = pad_sequence([item[2] for item in batch],
                                         batch_first=True,
                                         padding_value=self.pad_index)
        return {"mantissa": mantissa_stack,
                "exponent": exponent_stack,
                "latex_token": latex_token_stack,
                "equation": [item[3] for item in batch],
                'trg_len': torch.tensor([len(item[2]) for item in batch])}

    def worker_init_fn(self, worker_id):
        self.worker_seeds[worker_id] = self.config.generator.seed + worker_id
        seed = self.worker_seeds[worker_id]
        rng = default_rng(seed)
        self.generators[worker_id] = self.generator_class(rng=rng)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # print(f"worker {worker_id} has seed: {seed}")

    def worker_init_fn_validation(self, worker_id):
        seed = self.val_seed + worker_id
        rng = default_rng(seed)
        self.generators[worker_id] = self.generator_class(rng=rng)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # print(f"validation worker {worker_id} has seed: {seed}")
        
    def get_train_loader(self):
        """return a dataloader over an infinite set of training data."""
        train_dataset = EquationDataset(
            data_source=self.create_sample,
        )
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  num_workers=self.config.num_workers,
                                  worker_init_fn=self.worker_init_fn,
                                )
        return train_loader

    def get_valid_loader(self):
        valid_dataset = EquationDataset(
                    data_source=[self.create_sample() for _ in range(self.config.val_samples)],
                )
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  num_workers=self.config.num_workers,
                                  worker_init_fn=self.worker_init_fn_validation,
                                  )
        return valid_loader


if __name__ == "__main__":
    from gpr.data.generators import RandomGenerator, PolynomialGenerator

    sympy_data = SymPySimpleDataModule(generator=PolynomialGenerator,
                                       config_path='config/default_config.yaml')
    train_loader = sympy_data.get_train_loader()
    valid_loader = sympy_data.get_valid_loader()

    print("Validation equations:")
    counter = 0
    for batch in valid_loader:
        print(f"Batch {counter} equations:")
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
