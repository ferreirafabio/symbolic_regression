import torch

from gpr.data.utils import tokenize_latex_to_char


class EquationDataset(torch.utils.data.Dataset):

    def __init__(self, data_source, generator):
        self.data_source = data_source
        self.generator = generator

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

    def __len__(self):
        if callable(self.data_source):
            return 2**31  # A large number to simulate an infinite dataset
        else:
            return len(self.data_source)

