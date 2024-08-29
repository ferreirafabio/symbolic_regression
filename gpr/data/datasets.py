import torch
import sympy as sp

from gpr.data.utils import tokenize_latex_to_char


class EquationDataset(torch.utils.data.Dataset):

    def __init__(self, data_source):
        self.data_source = data_source

    def __getitem__(self, idx):
        if callable(self.data_source):
            # Generate samples on-the-fly
            mantissa, exponent, expression = self.data_source()
        else:
            # Access pre-generated samples
            mantissa, exponent, expression = self.data_source[idx % len(self.data_source)]

        latex_token_indices = tokenize_latex_to_char(sp.latex(expression))
        token_tensor = torch.tensor(latex_token_indices, dtype=torch.uint8)

        return mantissa, exponent, token_tensor, expression

    def __len__(self):
        if callable(self.data_source):
            return 2**31  # A large number to simulate an infinite dataset
        else:
            return len(self.data_source)

