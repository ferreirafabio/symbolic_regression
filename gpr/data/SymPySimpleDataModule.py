import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gpr.data.sympy_equation_sampler import sample_and_evaluate, sample_random_polynomial_equation
from gpr.data.data_module_template import AbstractDataModule
import sympy as sp
import re
import pandas as pd


import string

# Include all lowercase and uppercase letters, digits, and some special characters
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + '+-=()[]{}^_*\\/,.;:!`\'"<>|&%$#@~?'

# Create a dictionary mapping each character to a unique index
token_to_index = {ch: idx for idx, ch in enumerate(characters)}

def tokenize_latex_to_char(latex_string):
    return [token_to_index[ch] for ch in latex_string if ch in token_to_index]


class PolynomialDataset(Dataset):
    def __init__(self, create_sample, num_variables, num_realizations):
        self.create_sample = create_sample
        self.num_variables = num_variables
        self.num_realizations = num_realizations

    def __getitem__(self, idx):
        eq, vars_in_eq, latex_eq = self.create_sample()
        data = sample_and_evaluate(eq, vars_in_eq, num_samples=self.num_realizations, real_numbers_realizations=True)
        
        latex_token_indices = tokenize_latex_to_char(latex_eq)
        token_tensor = torch.tensor(latex_token_indices, dtype=torch.long)

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

        return (mantissa_data, exponent_data, token_tensor)

    def __len__(self):
        return 2**31



class SymPySimpleDataModule(AbstractDataModule):
    def __init__(self, num_variables, num_realisations, val_samples, batch_size=16):
        self.batch_size = batch_size
        self.val_samples = val_samples
        self.num_variables = num_variables
        self.num_realisations = num_realisations
        self.validation_set = self.create_validation_set()
        super().__init__(num_variables, {}, {}, num_realisations, val_samples)

    def create_sample(self, rng=None):
        return sample_random_polynomial_equation(max_powers=3, max_vars=self.num_variables, max_terms=3, real_numbers_variables=True)

    def create_validation_set(self):
        validation_data = [self.create_sample() for _ in range(self.val_samples)]
        validation_dataset = PolynomialDataset(lambda idx: validation_data[idx % self.val_samples], num_variables=self.num_variables, num_realizations=1)  # Each validation sample as separate
        return DataLoader(validation_dataset, batch_size=self.batch_size, collate_fn=self.collator)
    
    def collator(self, batch):
        mantissa_stack = torch.stack([item[0] for item in batch])
        exponent_stack = torch.stack([item[1] for item in batch])
        latex_token_stack = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
        return (mantissa_stack, exponent_stack, latex_token_stack)

    def get_train_loader(self, num_workers=8):
        train_dataset = PolynomialDataset(create_sample=self.create_sample, num_variables=self.num_variables, num_realizations=self.num_realisations)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=num_workers)
        while True:
            for data in train_loader:
                yield data

    def get_valid_dataloader(self):
        return self.validation_set

if __name__ == "__main__":
    sympy_data = SymPySimpleDataModule(num_variables=3, num_realisations=100, val_samples=10, batch_size=32)
    train_loader = sympy_data.get_train_loader(num_workers=8)
    valid_loader = sympy_data.get_valid_dataloader()

    for batch in train_loader:
        # (batch_size, num_realizations, num_variables + 1)
        print(f"mantissa batch shape: {batch[0].shape}")
        print(f"exponent batch shape: {batch[1].shape}")
        print(f"latex token batch shape: {batch[2].shape}")
