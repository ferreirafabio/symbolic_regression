import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gpr.data.sympy_equation_sampler import sample_and_evaluate, sample_random_polynomial_equation
from gpr.data.data_module_template import AbstractDataModule
import sympy as sp
import re
import pandas as pd
from numpy.random import default_rng
import numpy as np

import string

# Include all lowercase and uppercase letters, digits, and some special characters
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + '+-=()[]{}^_*\\/,.;:!`\'"<>|&%$#@~?'

# Create a dictionary mapping each character to a unique index
token_to_index = {ch: idx for idx, ch in enumerate(characters)}

def tokenize_latex_to_char(latex_string):
    return [token_to_index[ch] for ch in latex_string if ch in token_to_index]


class PolynomialDataset(Dataset):
    def __init__(self, data_source, num_variables, num_realizations):
        self.data_source = data_source
        self.num_variables = num_variables
        self.num_realizations = num_realizations

    def __getitem__(self, idx):
        if callable(self.data_source):
            # Generate samples on-the-fly
            eq, vars_in_eq, latex_eq = self.data_source()
        else:
            # Access pre-generated samples
            eq, vars_in_eq, latex_eq = self.data_source[idx % len(self.data_source)]

        data = sample_and_evaluate(eq, vars_in_eq, num_samples=self.num_realizations, real_numbers_realizations=True)

        latex_token_indices = tokenize_latex_to_char(latex_eq)
        token_tensor = torch.tensor(latex_token_indices, dtype=torch.long)

        # TODO add EOS and SOS tokens (also in training/valid for shifting or shit in data loader)

        # Plus 1 for the 'y' variable
        mantissa_data = torch.zeros((self.num_realizations, self.num_variables + 1), dtype=torch.float32)
        exponent_data = torch.zeros((self.num_realizations, self.num_variables + 1), dtype=torch.float32)
        
        mantissa_idx, exponent_idx = 0, 0
        for key, values in data.to_dict(orient='list').items():
            if '_mantissa' in key:
                mantissa_data[:, mantissa_idx] = torch.tensor(values, dtype=torch.float32)
                mantissa_idx += 1
            elif '_exponent' in key:
                exponent_data[:, exponent_idx] = torch.tensor(values, dtype=torch.float32)
                exponent_idx += 1

        return (mantissa_data, exponent_data, token_tensor, eq, data)

    def __len__(self):
        if callable(self.data_source):
            return 2**31  # A large number to simulate an infinite dataset
        else:
            return len(self.data_source)



class SymPySimpleDataModule(AbstractDataModule):
    def __init__(self, num_variables, max_powers=2, max_terms=4, num_realisations=1000, val_samples=500, batch_size=32, real_numbers_variables=False, seed=1):
        self.rng = default_rng(seed=seed)
        self.batch_size = batch_size
        self.val_samples = val_samples
        self.num_variables = num_variables
        self.num_realisations = num_realisations
        self.max_terms = max_terms
        self.real_numbers_variables = real_numbers_variables
        self.max_powers = max_powers
        self.validation_set = self.create_validation_set()
        super().__init__(num_variables, {}, {}, num_realisations, val_samples)

        self.ignore_index = -100
        self.pad_index = 0

    @property
    def vocab_size(self):
        return len(characters)

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch


    def create_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(np.random.randint(0, 1000000))
        return sample_random_polynomial_equation(max_powers=self.max_powers,
                                                 max_vars=self.num_variables,
                                                 max_terms=self.max_terms,
                                                 real_numbers_variables=self.real_numbers_variables, rng=rng)

    def create_validation_set(self):
        validation_data = [self.create_sample(rng=self.rng) for _ in range(self.val_samples)]
        validation_dataset = PolynomialDataset(
            data_source=validation_data,
            num_variables=self.num_variables,
            num_realizations=self.num_realisations
        )
        return DataLoader(validation_dataset, batch_size=self.batch_size, collate_fn=self.collator)

    
    def collator(self, batch):
        mantissa_stack = torch.stack([item[0] for item in batch])
        exponent_stack = torch.stack([item[1] for item in batch])
        latex_token_stack = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=self.pad_index)
        return {"mantissa": mantissa_stack, "exponent": exponent_stack, "latex_token": latex_token_stack, "equation": [item[3] for item in batch], "realizations": [item[4] for item in batch], 'trg_len': torch.tensor([len(item[2]) for item in batch])}

    def get_train_loader(self, num_workers=8):
        train_dataset = PolynomialDataset(
            data_source=self.create_sample,  # Pass the function to generate samples on-the-fly
            num_variables=self.num_variables,
            num_realizations=self.num_realisations
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=num_workers)
        while True:
            for data in train_loader:
                yield data

    def get_valid_loader(self):
        return self.validation_set

if __name__ == "__main__":
    sympy_data = SymPySimpleDataModule(num_variables=4, num_realisations=10, val_samples=500, batch_size=32)
    train_loader = sympy_data.get_train_loader()
    valid_loader = sympy_data.get_valid_loader()

    for batch in valid_loader:
        # (batch_size, num_realizations, num_variables + 1)
        print(f"mantissa batch shape: {batch['mantissa'].shape}")
        print(f"exponent batch shape: {batch['exponent'].shape}")
        print(f"latex token batch shape: {batch['latex_token'].shape}")
        
        print(f"equation sanity check1: {batch['equation'][0]}")
        print(f"equation sanity check2: {batch['equation'][1]}")
        
        print(f"realization sanity check1: {batch['realizations'][0]}")
        print(f"realization sanity check1: {batch['realizations'][1]}")
        break
