import sympy as sp
import random
import pandas as pd
import math
import numpy as np
import torch
from sympy import symbols, lambdify, Eq
from sympy.parsing.sympy_parser import parse_expr


def sample_random_polynomial_equation(max_powers, max_vars, max_terms, real_numbers_variables=False, rng=None):
    # If rng is not provided, create a default random generator using numpy
    rng = rng if rng is not None else np.random.default_rng()
    # rng = rng or np.random.default_rng()
    
    variables = [sp.symbols(f'x{i}') for i in range(1, max_vars + 1)]
    polynomial = 0
    num_terms = rng.integers(1, max_terms + 1)

    # Keep generating terms until a non-zero polynomial is obtained
    while polynomial == 0:
        polynomial = 0
        for _ in range(num_terms):
            num_vars_in_term = rng.integers(1, len(variables) + 1)  # exclusive upper bound
            term_variables = variables[:num_vars_in_term]

            if real_numbers_variables:
                term = rng.uniform(-10, 10)
            else:
                term = rng.integers(-10, 10)

            for var in term_variables:
                exponent = rng.integers(1, max_powers + 1)
                term *= var ** exponent
            polynomial += term

    y = sp.symbols('y')
    equation = sp.Eq(y, polynomial)
    latex_equation = sp.latex(equation)

    return equation, variables, latex_equation

def sample_and_evaluate(eq, vars_in_eq, num_samples, real_numbers_realizations=True):
    used_vars = [var for var in vars_in_eq if var in eq.rhs.free_symbols]
    if real_numbers_realizations:
        random_realizations = torch.FloatTensor(num_samples, len(used_vars)).uniform_(-10, 10)
    else:
        random_realizations = torch.randint(-10, 10, (num_samples, len(used_vars)), dtype=torch.float32)

    equation_func = lambdify(used_vars, eq.rhs, modules="numpy")

    computed_y_values_np = equation_func(*[random_realizations[:, idx].numpy() for idx in range(len(used_vars))])
    computed_y_values = torch.tensor(computed_y_values_np, dtype=torch.float32)

    var_mantissa, var_exponent = random_realizations.frexp()
    y_mantissa, y_exponent = computed_y_values.frexp()

    mantissa = torch.cat([y_mantissa.unsqueeze(1), var_mantissa], dim=1)
    exponent = torch.cat([y_exponent.unsqueeze(1), var_exponent], dim=1)

    return mantissa, exponent


if __name__ == "__main__":
    # Example usage:
    vars_in_eq = symbols("x y z")
    eq_str = "y = x + 2 * z"
    eq = Eq(*[parse_expr(expr) for expr in eq_str.split("=")])
    num_samples = 400
    mantissa_tensor, exponent_tensor = sample_and_evaluate(eq, vars_in_eq, num_samples)
    print(mantissa_tensor)
    print(exponent_tensor)

