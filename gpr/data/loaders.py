import sympy as sp
import pandas as pd
import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng

from gpr.data.abstract import AbstractDataModule
from gpr.data.generators import SimpleGenerator, PolynomialGenerator
from gpr.data.datasets import characters, SimpleDataset
from gpr.utils.configuration import Config


class SymPySimpleDataModule(AbstractDataModule):
    def __init__(self, generator, config_path):
        global_config = Config(config_file=config_path)
        self.config = global_config.dataloader
        self.generator = generator(rng=default_rng(self.config.generator.seed))
        self.rng = self.generator.rng

        self.ignore_index = -100
        self.pad_index = 0
        self.validation_set = self.create_validation_set()

    def get_vocab(self):
        return characters

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch

    def create_sample(self, rng=None):
        if rng is not None:
            self.generator.rng = rng
        return self.generator(**self.config.generator)

    def create_validation_set(self):
        validation_data = [self.create_sample(rng=self.rng) for _ in range(self.config.val_samples)]
        validation_dataset = SimpleDataset(
            data_source=validation_data,
            generator=self.generator
        )

    def collator(self, batch):
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

    def get_train_loader(self):
        train_dataset = SimpleDataset(
            data_source=self.create_sample, # Pass the function to generate samples on-the-fly
            generator=self.generator
        )
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  num_workers=self.config.num_workers)
        return train_loader

    def get_valid_loader(self):
        valid_loader = DataLoader(self.validation_set,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  num_workers=self.config.num_workers)
        return valid_loader

if __name__ == "__main__":
    sympy_data = SymPySimpleDataModule(generator=PolynomialGenerator,
                                       config_path='config/default_config.yaml')
    train_loader = sympy_data.get_train_loader()
    valid_loader = sympy_data.get_valid_loader()

    for batch in train_loader:
        print(f"mantissa batch shape: {batch['mantissa'].shape}")
