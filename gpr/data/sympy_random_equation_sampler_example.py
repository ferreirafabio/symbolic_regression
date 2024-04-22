import sympy as sp
import random
import pandas as pd


def sample_random_polynomial_equation(max_powers, max_vars, max_terms, real_numbers_variables=False):
    # Define the independent variables
    variables = [sp.symbols(f'x{i}') for i in range(1, max_vars + 1)]
    polynomial = 0
    num_terms = random.randint(1, max_terms)

    # Keep generating terms until a non-zero polynomial is obtained
    while polynomial == 0:
        polynomial = 0
        for _ in range(num_terms):
            # Number of variables to include in this term
            num_vars_in_term = random.randint(1, len(variables))
            
            # Select variables sequentially starting from x1
            term_variables = variables[:num_vars_in_term]

            term = random.uniform(-10, 10) if real_numbers_variables else random.randint(-10, 10)
            for var in term_variables:
                exponent = random.randint(1, max_powers)
                term *= var ** exponent
            polynomial += term

    y = sp.symbols('y')
    equation = sp.Eq(y, polynomial)

    return equation, variables



def sample_and_evaluate(eq, vars_in_eq, num_samples=1000, real_numbers_realizations=False):
    data = []
    for _ in range(num_samples):
        values_for_x = {var: 0 for var in vars_in_eq}
        
        # Update values for only the variables present in the equation
        used_vars = [var for var in vars_in_eq if var in eq.rhs.free_symbols]
        for var in used_vars:
            values_for_x[var] = random.uniform(-10, 10) if real_numbers_realizations else random.randint(-10, 10)

        computed_y_value = eq.rhs.subs(values_for_x).evalf()
        data.append({**values_for_x, 'y': float(f"{computed_y_value:.1f}")})
    return pd.DataFrame(data)

if __name__ == "__main__":
    equation, variables_in_equation = sample_random_polynomial_equation(max_powers=3, max_vars=4, max_terms=3, real_numbers_variables=False)
    print("Generated polynomial equation:", equation)

    # Sample realizations and evaluate the equation
    result_table = sample_and_evaluate(equation, variables_in_equation, num_samples=1000, real_numbers_realizations=True)
    print(result_table.head())
