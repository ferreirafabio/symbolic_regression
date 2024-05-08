import string
import torch

from gpr.data.abstract import AbstractDataset
from gpr.data.sympy_equation_sampler import sample_and_evaluate


# Include all lowercase and uppercase letters, digits, and some special characters
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + '+-=()[]{}^_*\\/,.;:!`\'"<>|&%$#@~?'

# Create a dictionary mapping each character to a unique index
token_to_index = {ch: idx for idx, ch in enumerate(characters)}

def tokenize_latex_to_char(latex_string):
    return [token_to_index[ch] for ch in latex_string if ch in token_to_index]


class PolynomialDataset(AbstractDataset):
    def __init__(self, data_source, num_variables, num_realizations):
        self.num_variables = num_variables
        self.num_realizations = num_realizations
        super().__init__(data_source, generator=None)

    def __getitem__(self, idx):
        if callable(self.data_source):
            # Generate samples on-the-fly
            eq, vars_in_eq, latex_eq = self.data_source()
        else:
            # Access pre-generated samples
            eq, vars_in_eq, latex_eq = self.data_source[idx % len(self.data_source)]

        mantissa, exponent = sample_and_evaluate(eq, vars_in_eq, num_samples=self.num_realizations, real_numbers_realizations=True)

        latex_token_indices = tokenize_latex_to_char(latex_eq)
        token_tensor = torch.tensor(latex_token_indices, dtype=torch.long)

        # TODO add EOS and SOS tokens (also in training/valid for shifting or shit in data loader)

        return mantissa, exponent, token_tensor, eq


class SimpleDataset(AbstractDataset):

    def __getitem__(self, idx):
        if callable(self.data_source):
            # Generate samples on-the-fly
            mantissa, exponent = self.data_source()
        else:
            # Access pre-generated samples
            mantissa, exponent = self.data_source[idx % len(self.data_source)]

        latex_token_indices = tokenize_latex_to_char(self.generator.expression_latex)
        token_tensor = torch.tensor(latex_token_indices, dtype=torch.long)

        return mantissa, exponent, token_tensor, self.generator.expression
