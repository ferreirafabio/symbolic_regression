import sympy as sp
import random
import pandas as pd
import math
import numpy as np


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



def sample_and_evaluate(eq, vars_in_eq, num_samples=1000, real_numbers_realizations=False):
    data = []
    for _ in range(num_samples):
        values_for_x = {var: 0 for var in vars_in_eq}
        
        # Update values for only the variables present in the equation
        decomposition = {}
        used_vars = [var for var in vars_in_eq if var in eq.rhs.free_symbols]
        for var in used_vars:
            realization = random.uniform(-10, 10) if real_numbers_realizations else random.randint(-10, 10)

            mantissa, exponent = math.frexp(realization)
            values_for_x[var] = realization
            decomposition[f"{var}_mantissa"] = mantissa
            decomposition[f"{var}_exponent"] = exponent

        computed_y_value = eq.rhs.subs(values_for_x).evalf()
        y_mantissa, y_exponent = math.frexp(computed_y_value)
        data_point = {
            #**values_for_x, 
            #'y': float(f"{computed_y_value:.1f}"), 
            **decomposition,
            'y_mantissa': y_mantissa, 
            'y_exponent': y_exponent,
        }
        data.append(data_point)
    return pd.DataFrame(data)

if __name__ == "__main__":
    equation, variables_in_equation, latex_equation = sample_random_polynomial_equation(max_powers=3, max_vars=4, max_terms=3, real_numbers_variables=False)
    print("Generated polynomial equation:", equation)
    print("latex equation:", latex_equation)

    # Sample realizations and evaluate the equation
    result_table = sample_and_evaluate(equation, variables_in_equation, num_samples=1000, real_numbers_realizations=True)
    print(result_table.head())
