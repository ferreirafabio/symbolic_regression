import os
import pathlib

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
from gpr.data.utils import tokenize_latex_to_char



# TODO save as pyarrow
# TODO dedub
# TODO valid/train sim
# TODO parallel generation
# TODO junked dumping
# TODO save fp32 and int16


class CreateDataset(object):
    def __init__(self, generator, config_path, exp_folder):
        global_config = Config(config_file=config_path)
        self.config = global_config.dataloader
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed  # base seed for training set
        self.val_seed = self.seed + 1000  # base seed for validation set

        self.generator_class = generator
        self.generators = [None] * self.config.num_workers

        self.worker_seeds[0] = self.config.generator.seed + 0
        seed = self.worker_seeds[0]
        rng = default_rng(seed)
        self.generators[0] = self.generator_class(rng=rng)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)



        rng = default_rng(self.seed)
        self.generator = self.generator_class(rng=rng)

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





    def create_sample(self):
        sample = self.generator(**self.config.generator)

        return sample


    def create_set(self):
        dataset = []
        for _ in range(self.config.val_samples):
            mantissa, exponent, expression = self.create_sample()
            self.equation_logger.info(f"Validation equation: {expression}")
            latex_expression = sp.latex(expression)
            latex_token_indices = tokenize_latex_to_char(latex_expression)
            token_tensor = torch.tensor(latex_token_indices, dtype=torch.long)
            dataset.append((mantissa, exponent, token_tensor))

        return dataset


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
                'trg_len': torch.tensor([len(item[2]) for item in batch])}




    def get_valid_loader(self):
        """return a dataloader over an infinite set of training data."""
        train_dataset = self.create_set()

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.batch_size,
                                  collate_fn=self.collator,
                                  )
        return train_loader



if __name__ == "__main__":
    from gpr.data.generators import RandomGenerator, PolynomialGenerator

    sympy_data = CreateDataset(generator=PolynomialGenerator,
                                       config_path='config/data_config.yaml', exp_folder=pathlib.Path('.'))
    valid_loader = sympy_data.get_valid_loader()

    print("Validation equations:")
    counter = 0
    for batch in valid_loader:
        print(f"Batch {counter} equations:")
        for equation in batch['latex_token']:
            print(equation)
        counter += 1
        if counter == 5:
            break


