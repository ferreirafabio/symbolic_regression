import sympy as sp
import pandas as pd
import numpy as np
import torch
import yaml
import random
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
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed # base seed for training set
        self.val_seed = self.seed + 1000  # base seed for validation set
        self.generator_class = generator
        self.generators = [None] * self.config.num_workers

        self.ignore_index = -100
        self.pad_index = 0

    def get_vocab(self):
        return characters


    def indices_to_string(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([characters[i] for i in indices])

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
