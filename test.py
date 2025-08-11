import sympy as sp
from sympy.parsing.latex import parse_latex
import re

def clean_variable_names(equation):
    """Replace subscripted variable names with non-subscripted names."""
    lhs, rhs = equation.lhs, equation.rhs
    var_map = {}
    counter = 1

    # Sort variables to ensure consistent ordering
    sorted_vars = sorted(rhs.free_symbols, key=lambda v: str(v))

    for var in sorted_vars:
        var_str = str(var)
        if var_str == 'x':
            var_map[var] = sp.Symbol('x')
        elif var_str.startswith('x_'):
            # For variables like x_{1}, x_{2}, etc., use x1, x2, etc.
            new_var = sp.Symbol(f'x{var_str[2:].strip("{}")}')
            var_map[var] = new_var
        else:
            # For any other variables, use x{counter}
            new_var = sp.Symbol(f'x{counter}')
            var_map[var] = new_var
            counter += 1

    new_rhs = rhs.subs(var_map)
    return sp.Eq(lhs, new_rhs), var_map

def parse_and_clean_latex(latex_string):
    # Add explicit multiplication symbol * before parentheses
    latex_string = re.sub(r'(\w+)(\()', r'\1*\2', latex_string)
    
    # Parse the modified LaTeX string
    parsed_eq = parse_latex(latex_string)
    
    # Clean the equation
    cleaned_eq, var_map = clean_variable_names(parsed_eq)
    
    return parsed_eq, cleaned_eq, var_map

# LaTeX string
latex_string = r"y=x_{1}x_{2}\left(-3.89x_{1}x_{2}+6.22\right)"

# Parse and clean the LaTeX string
original_eq, cleaned_eq, var_map = parse_and_clean_latex(latex_string)

print("Original equation:")
print(original_eq)

print("\nFree symbols:")
print(original_eq.free_symbols)

print("\nCleaned equation:")
print(cleaned_eq)

print("\nVariable mapping:")
for old_var, new_var in var_map.items():
    print(f"{old_var} -> {new_var}")
