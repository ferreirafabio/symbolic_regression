import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gpr.data.sympy_equation_sampler import sample_and_evaluate, sample_random_polynomial_equation
from gpr.data.data_module_template import AbstractDataModule
import sympy as sp
import random
import pandas as pd




class PolynomialDataset(Dataset):
    def __init__(self, create_sample, num_variables, num_realizations):
        self.create_sample = create_sample
        self.num_variables = num_variables
        self.num_realizations = num_realizations

    def __getitem__(self, idx):
        eq, vars_in_eq = self.create_sample()
        data = sample_and_evaluate(eq, vars_in_eq, num_samples=self.num_realizations, real_numbers_realizations=True)

        num_cols = (self.num_variables * 2) + 2  # for each x we have mantissa + exponent + mantissa and exponent for y
        tensor_data = torch.zeros((self.num_realizations, num_cols), dtype=torch.float32)
        start_idx = 0
        # assign data to get shape (num_realisations, num_cols) with zero padding for missing values
        for key, values in data.to_dict(orient='list').items():
            if '_mantissa' in key or '_exponent' in key or 'y_mantissa' == key or 'y_exponent' == key:
                tensor_data[:, start_idx] = torch.tensor(values, dtype=torch.float32)
                start_idx += 1
        return tensor_data

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
        return torch.stack(batch)

    def get_infinite_train_loader(self, num_workers=0):
        train_dataset = PolynomialDataset(create_sample=self.create_sample, num_variables=self.num_variables, num_realizations=self.num_realisations)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=num_workers)
        while True:
            for data in train_loader:
                yield data

    def get_valid_dataloader(self):
        return self.validation_set

if __name__ == "__main__":
    sympy_data = SymPySimpleDataModule(num_variables=3, num_realisations=100, val_samples=10, batch_size=4)
    train_loader = sympy_data.get_infinite_train_loader()
    valid_loader = sympy_data.get_valid_dataloader()

    for batch in train_loader:
        print(batch.shape)
