import sympy as sp
import pandas as pd
import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng

from gpr.data.datasets import EquationDataset
from gpr.data.utils import characters
from gpr.utils.configuration import Config


class SymPySimpleDataModule(object):
    def __init__(self, generator, config_path):
        global_config = Config(config_file=config_path)
        self.config = global_config.dataloader
        self.seed = self.config.generator.seed
        self.generator = generator(rng=default_rng(self.seed))
        self.rng = self.generator.rng

        self.ignore_index = -100
        self.pad_index = 0
        self.validation_set = self.create_validation_set()

    def get_vocab(self):
        return characters

    @property
    def vocab_size(self):
        return len(self.get_vocab())

    def latex_equation_to_function(self, latex_equation):
        #TODO
        pass

    def check_if_latex_equation_is_valid(self):
        #TODO
        return True

    def batch_to_device(self, batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch

    def create_sample(self, seed=None):
        """Return a tbl of n inference samples of the equation, and the target
        token sequnce of the latex equation."""
        if seed is not None:
            self.generator.rng = default_rng(seed=seed)
        return self.generator(**self.config.generator)

    def create_validation_set(self):
        validation_data = [self.create_sample() for _ in range(self.config.val_samples)]
        validation_dataset = EquationDataset(
            data_source=validation_data,
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

    def get_train_loader(self):
        """return a dataloader over an infinite set of training data."""
        train_dataset = EquationDataset(
            data_source=self.create_sample, # Pass the function to generate samples on-the-fly
        )
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  num_workers=self.config.num_workers)
        return train_loader

    def get_valid_loader(self):
        """return a dataloader over a finite set of validation data, which was
        created in the create_validation_set method."""
        valid_loader = DataLoader(self.validation_set,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  num_workers=self.config.num_workers)
        return valid_loader


if __name__ == "__main__":
    from gpr.data.generators import RandomGenerator, PolynomialGenerator

    sympy_data = SymPySimpleDataModule(generator=PolynomialGenerator,
                                       config_path='config/default_config.yaml')
    train_loader = sympy_data.get_train_loader()
    valid_loader = sympy_data.get_valid_loader()

    print("Validation equations:")
    for batch_valid in valid_loader:
        print(f"equation batch {batch_valid['equation']}")

    print("Training equations:")
    counter = 0
    for batch in train_loader:
        print(f"equation batch {batch_valid['equation']}")
        #print(f"equation batch {batch_valid['mantissa']}")
        counter += 1
        if counter == 5:
            break
