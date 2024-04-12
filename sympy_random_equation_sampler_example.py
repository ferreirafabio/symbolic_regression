import sympy as sp
from sympy import latex, solve
sp.init_printing()
import random


def generate_random_equation(max_powers, max_vars, max_terms):
    # create symbols
    variables = [sp.symbols(f'x{i}') for i in range(1, max_vars + 1)]
    equation = 0

    # adjust to start from a different minimum of # terms
    num_terms = random.randint(1, max_terms)

    for _ in range(num_terms):
        term_variables = random.sample(variables, random.randint(1, len(variables)))

        term = random.randint(-10, 10)  # coefficient
        for var in term_variables:
            exponent = random.randint(1, max_powers)  # exponent
            term *= var ** exponent

        equation += term

    return equation


eq = generate_random_equation(max_powers=3, max_vars=3, max_terms=2)
print(solve(eq))
print(latex(eq))
