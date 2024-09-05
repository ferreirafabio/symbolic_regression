import torch
import sympy as sp
import random
import numpy as np
import pandas as pd

from glob import glob
from functools import partial

from gpr.data.abstract import AbstractGenerator
from gpr.data.generators.base_generator import BaseGenerator
from gpr.data.utils import format_floats_recursive, get_sym_model, read_pmlb_files


class PMLBGenerator(BaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.index = 0
        self.raw_df = None # data not loaded yet

    @classmethod
    def load_data(cls, *args, groups=['feynman', 'strogatz', 'black-box'],
                  **kwargs):
        generator = cls(*args, **kwargs)
        generator.load_pmlb_data(groups=groups)
        return generator

    def load_pmlb_data(self, datadir='pmlb/datasets/', groups=['feynman']):
        frames = []
        for i, f in enumerate(glob(datadir+'/*/*.tsv.gz')):
            group = 'feynman' if 'feynman' in f else 'strogatz' if 'strogatz' in f else 'black-box'
            if group not in groups:
                continue

            df = pd.read_csv(f, sep='\t')
            features, labels, feature_names = read_pmlb_files(
                f, use_dataframe=True, sep='\t'
            )
            expression = get_sym_model(f, return_str=False) if group != 'black-box' else ''
            frames.append(dict(
                name=f.split('/')[-1][:-7],
                nsamples = df.shape[0],
                nvars = df.shape[1],
                npoints = df.shape[0]*df.shape[1],
                Group=group,
                X = features,
                Y = labels,
                X_names = feature_names,
                Expression = expression
            ))

        df = pd.DataFrame.from_records(frames)
        self.raw_df = df

    def __iter__(self):
        return self

    def __call__(self, num_realizations: int=10, nan_threshold: float=0.1,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            x, y, expression = next(self)
            x, y = x[:num_realizations, ...], y[:num_realizations, ...]
            skip, is_nan = self.check_nan_inf(y, nan_threshold)
            if skip:
                continue

            m, e = BaseGenerator.get_mantissa_exp(x, y)
            return m, e, expression, is_nan

    def __next__(self):
        if self.index < len(self.raw_df):
            self.x_data = self.raw_df.X.values[self.index].to_numpy().astype('float32')
            self.y_data = self.raw_df.Y.values[self.index].astype('float32')
            self.generate_equation(self.index)

            self.index += 1
            return self.x_data, self.y_data, self.expression
        else:
            raise StopIteration('All equations have been consumed.')

    def generate_equation(self, index: int, **kwargs) -> None:
        self.expression = self._generate_random_expression(index, **kwargs)
        self.expression_str = str(self.expression.rhs)
        self.expression_latex = sp.latex(self.expression)

        # in case we want to evaluate the equations using other realizations
        self.equation = sp.lambdify(list(self.expression.rhs.free_symbols),
                                    self.expression.rhs,
                                    modules="numpy")

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, index: int, **kwargs) -> sp.Eq:
        return self.raw_df.Expression.values[index]


FeynmanGenerator = partial(PMLBGenerator.load_data, groups=['feynman'])
StrogatzGenerator = partial(PMLBGenerator.load_data, groups=['strogatz'])
BlackboxGenerator = partial(PMLBGenerator.load_data, groups=['black-box'])


if __name__ == '__main__':
    from gpr.data.utils import tokenize_latex_to_char, token_to_index
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    def detokenize_indices_to_latex(indices):
        """
        Convert a list of token indices back to a LaTeX string.
        """
        return ''.join(index_to_token[idx] for idx in indices)

    generator = FeynmanGenerator(verbose=True)

    # Define the parameters for the equation generation
    params = {
        "num_variables": 6,
        "num_realizations": 256,  # We generate one realization per loop iteration
        "max_terms": 4,
        "max_powers": 4,
        "use_constants": True,
        "allowed_operations": ["+", "-", "*", "/", "sin", "cos", "asin", "acos", "sqrt"],
        "keep_graph": False,
        "keep_data": False,
        "use_epsilon": True,
        "max_const_exponent": 2,
        "real_const_decimal_places": 0,
        "real_constants_min": -5,
        "real_constants_max": 5,
        "nan_threshold": 0.5,
        "max_depth": 4, # 0-indexed depth
        "nesting_probability": 0.8,
        "unary_operation_probability": 0.2,
        "sample_interval": [-1, 1],
        "exponent_probability": 0.1,
    }

    # Generate and print 5 different equations
    for i in range(10):
        mantissa, exponent, expression, is_nan = generator(**params)
        print(f"Equation {i+1}: {expression} expression contains NaNs: {is_nan}")

        def get_constants(equation):
            return list(equation.atoms(sp.Number))

        constants = get_constants(expression)
        print(f"Constants: {constants}")

