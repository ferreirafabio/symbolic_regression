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

from gpr.data.utils import all_tokens, token_to_index
from gpr.data.data_creator import get_base_name


# TODO multi gpu, multi node loading
# TODO samller realization as in data arraw file
# TODO speed test



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
    def __init__(self, global_config, logger):
        self.config = global_config.dataloader
        self.worker_seeds = np.zeros(self.config.num_workers, dtype=int)
        self.seed = self.config.generator.seed # base seed for training set
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
                if key == 'mantissa':
                    tensor_data = tensor_data[:num_realizations, :]
                    tensor_data = tensor_data.float().round(decimals=self.real_const_decimal_places).to(torch.float16)

                elif key == 'exponent':
                    tensor_data = tensor_data[:num_realizations, :]
                    tensor_data = tensor_data.to(torch.int8)

                batch[key].append(tensor_data.t())  # Transpose for PyTorch

            # batch['latex_expression'].append(sample['latex_expression'].to_pylist()[0])

        mantissa_stack = pad_sequence(batch['mantissa'],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1)
        exponent_stack = pad_sequence(batch['exponent'],
                                      batch_first=True,
                                      padding_value=self.pad_index).transpose(2,1).to(mantissa_stack.dtype)

        input_seq = [torch.cat([torch.tensor([self.soe_index], dtype=torch.uint8), tokens]) for tokens in batch['token_tensor']]
        input_seq_stack = pad_sequence(input_seq, batch_first=True, padding_value=self.pad_index)

        target_seq = [torch.cat([tokens, torch.tensor([self.eoe_index], dtype=torch.uint8)]) for tokens in batch['token_tensor']]
        target_seq_stack = pad_sequence(target_seq, batch_first=True, padding_value=self.ignore_index)

        return {"mantissa": mantissa_stack,
                "exponent": exponent_stack,
                "in_equation": input_seq_stack,
                "trg_equation": target_seq_stack,
                # "latex_equation": batch['latex_expression'],
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

        self.logger.info(f"DataLoader for {set_name} successfully loaded from {file_dir}.")

        train_set_size = dataset.num_record_batches
        self.logger.info(f"Number of {set_name} samples: {train_set_size}")

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
        # if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.info(f"Inspecting sample batch for {set_name} dataset")

        # check content in dataloader
        sample_batch = next(iter(data_loader))

        # self.logger.info(f"Let's see what's in the data. Sample batch keys: {sample_batch.keys()}")
        #
        # # Log the first few values of each field in the sample batch
        # for key, value in sample_batch.items():
        #     if isinstance(value, torch.Tensor):
        #         self.logger.info(f"Sample values for {key}: Shape {value.shape}, First few values: {value[:5]}")
        #     elif isinstance(value, list):
        #         self.logger.info(f"Sample values for {key}: First few values: {value[:5]}")
        #     else:
        #         self.logger.info(f"Sample values for {key}: {value}")

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
