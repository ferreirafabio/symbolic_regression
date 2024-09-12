import sympy as sp
import numpy as np
import yaml
from gpr.data.generators.polynomial_generator import PolynomialGenerator
from gpr.data.utils import format_floats_recursive


def test_polynomial_generator(config, total_equations=50):
    generator = PolynomialGenerator(verbose=False)
    
    params = config['dataloader']['generator']

    error_count = 0
    nan_inf_count = 0
    lambdify_error_count = 0
    empty_constants_count = 0
    error_types = {}
    operation_family_counts = {family: 0 for family in generator.operation_families}

    for i in range(total_equations):
        try:
            result = generator(**params)
            if result is None:
                print(f"Equation {i+1}: Generation failed")
                continue
            mantissa, exponent, expression, is_nan = result
            print(f"\nEquation {i+1}: {expression.rhs}")
            print(f"Expression contains NaNs: {is_nan}")
        except Exception as e:
            print(f"Error generating equation {i+1}: {e}")
            continue

        # Count operation families used
        for family, ops in generator.operation_families.items():
            if any(str(op) in str(expression.rhs) for op in ops):
                operation_family_counts[family] += 1

        variables = list(expression.rhs.free_symbols)
        # print(f"Before lambdification: {expression.rhs}")
        
        try:
            lambda_func = sp.lambdify(variables, expression.rhs, modules=['numpy', 'sympy'])
            # print(f"After lambdification: {lambda_func}")
            # print("Lambdification successful")
        except Exception as e:
            print(f"Lambdification failed: {str(e)}")
            lambdify_error_count += 1
            error_type = type(e).__name__
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(str(e))
            continue

        constants = list(expression.rhs.atoms(sp.Number))
        print(f"Constants: {constants}")
        if not constants:
            print("Warning: No constants in the equation")
            empty_constants_count += 1

        num_realizations = params['num_realizations']
        num_variables = len(variables)
        kmax = params.get('kmax', 5)
        
        var_values = generator.sample_from_mixture(
            num_samples=num_realizations,
            num_dimensions=num_variables,
            kmax=kmax,
            mean=0,
            var=1
        )
        # print(var_values)
        # Clip the input data
        clipped_var_values = generator.clip_x_data(var_values, generator.operation_families, expression.rhs)
        # print(clipped_var_values)

        try:
            with np.errstate(all='raise'):
                y_values = lambda_func(*[clipped_var_values[:, i] for i in range(num_variables)])
            
            print(f"Realized y values (first 5): {y_values[:5]}")
            has_nan = np.any(np.isnan(y_values))
            has_inf = np.any(np.isinf(y_values))
            print(f"Any Inf/NaN in y values: {has_inf}/{has_nan}")
            
            if has_nan or has_inf:
                nan_inf_count += 1

        except Exception as e:
            print(f"Error occurred while evaluating the function: {str(e)}")
            error_count += 1
            error_type = type(e).__name__
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(str(e))

    print(f"\nTotal equations: {total_equations}")
    print(f"Equations with errors: {error_count} ({error_count/total_equations*100:.2f}%)")
    print(f"Equations with NaN or Inf values: {nan_inf_count} ({nan_inf_count/total_equations*100:.2f}%)")
    print(f"Equations with lambdification errors: {lambdify_error_count} ({lambdify_error_count/total_equations*100:.2f}%)")
    print(f"Equations with no constants: {empty_constants_count} ({empty_constants_count/total_equations*100:.2f}%)")
    print(f"\nNumber of unique error types: {len(error_types)}")
    print("\nError types and messages:")
    for error_type, messages in error_types.items():
        print(f"{error_type} (Count: {len(messages)}, {len(messages)/total_equations*100:.2f}%):")
        for message in set(messages):  # Use set to remove duplicates
            print(f"  - {message}")

    print("\nOperation family usage:")
    for family, count in operation_family_counts.items():
        percentage = count / total_equations * 100
        print(f"{family}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":

    config_path = 'config/feynman_arc_config.yaml'  

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    test_polynomial_generator(config=config, total_equations=50)