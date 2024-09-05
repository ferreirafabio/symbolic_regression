# import string

# # Include all lowercase and uppercase letters, digits, and some special characters
# characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + '+-=()[]{}^_*\\/,.;:!`\'"<>|&%$#@~?'

# # Create a dictionary mapping each character to a unique index
# token_to_index = {ch: idx for idx, ch in enumerate(characters)}

# def tokenize_latex_to_char(latex_string):
#     return [token_to_index[ch] for ch in latex_string if ch in token_to_index]

import re
import sympy as sp
import numpy as np
import pandas as pd

from yaml import load, Loader
from sympy import Symbol, simplify, factor, Float, preorder_traversal, Integer
from sympy.parsing.sympy_parser import parse_expr


# Define LaTeX commands
latex_commands = [
    # Greek letters (commonly used in math)
    r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\zeta', r'\eta', r'\theta', r'\iota', r'\kappa',
    r'\lambda', r'\mu', r'\nu', r'\xi', r'\pi', r'\rho', r'\sigma', r'\tau', r'\upsilon', r'\phi', r'\chi',
    r'\psi', r'\omega',
    # Mathematical operators and symbols
    r'\sum', r'\prod', r'\int', r'\frac', r'\sqrt', r'\lim', r'\inf', r'\sup', r'\max', r'\min',
    r'\to', r'\rightarrow', r'\leftarrow', r'\leftrightarrow',
    # Parentheses and brackets
    r'\left', r'\right', r'\big', r'\Big', r'\bigg', r'\Bigg',
    # Trigonometric and other functions
    r'\sin', r'\cos', r'\tan', r'\log', r'\ln', r'\exp', r'\Mod', r'\Abs',
    # Misc mathematical symbols
    r'\infty', r'\pm', r'\mp', r'\times', r'\div', r'\cdot', r'\equiv', r'\neq', r'\approx', r'\propto',
]

# Special characters
special_chars = ['x', 'y', '_', '^', '{', '}', '[', ']', '(', ')', '+', '-', '=', '<', '>', '/', '*', ',', '.']

# Digits
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# special tokens
special_tokens = ['<SOE>', '<EOE>', '<PAD>'] # Start of equation, end of equation, padding

# Combine all tokens
all_tokens = latex_commands + special_chars + digits + special_tokens

# Create a dictionary mapping each token to a unique index
token_to_index = {token: idx for idx, token in enumerate(all_tokens)}

def tokenize_latex_to_char(latex_string):
    tokens = []
    i = 0
    while i < len(latex_string):
        # Check for LaTeX commands
        for cmd in latex_commands:
            if latex_string.startswith(cmd, i):
                tokens.append(token_to_index[cmd])
                i += len(cmd)
                break
        else:
            # Check for variables (x followed by number)
            if latex_string[i] == 'x' and i+1 < len(latex_string) and latex_string[i+1].isdigit():
                tokens.append(token_to_index['x'])
                i += 1
                # Tokenize the number following 'x'
                number_start = i
                while i < len(latex_string) and latex_string[i].isdigit():
                    i += 1
                for digit in latex_string[number_start:i]:
                    tokens.append(token_to_index[digit])
            # Check for scientific notation (e.g., 1.0e-9)
            elif latex_string[i].isdigit() or (latex_string[i] == '.' and i + 1 < len(latex_string) and latex_string[i + 1].isdigit()):
                number_start = i
                while i < len(latex_string) and (latex_string[i].isdigit() or latex_string[i] in ['.', 'e', '+', '-']):
                    if latex_string[i] == 'e':
                        tokens.append(token_to_index['e'])
                    elif latex_string[i] in ['+', '-']:
                        tokens.append(token_to_index[latex_string[i]])
                    else:
                        tokens.append(token_to_index[latex_string[i]])
                    i += 1
            # Check for numbers
            elif latex_string[i].isdigit():
                number_start = i
                while i < len(latex_string) and latex_string[i].isdigit():
                    i += 1
                for digit in latex_string[number_start:i]:
                    tokens.append(token_to_index[digit])
            # Check for special characters
            elif latex_string[i] in special_chars:
                tokens.append(token_to_index[latex_string[i]])
                i += 1
            else:
                # Skip unknown characters
                i += 1
    return tokens


def format_floats_recursive(expr, decimal_places):
    """
    Recursively ensures that all float values in the expression have exactly `decimal_places` digits.
    """
    if isinstance(expr, sp.Float):
        return sp.Float(round(expr, decimal_places), decimal_places)
    elif expr.is_Atom:
        return expr
    else:
        return expr.func(*[format_floats_recursive(arg, decimal_places) for arg in expr.args])

def read_pmlb_files(filename, label='target', use_dataframe=True, sep=None):

    if filename.endswith('gz'):
        compression = 'gzip'
    else:
        compression = None

    print('compression:',compression)
    print('filename:',filename)

    input_data = pd.read_csv(filename, sep=sep, compression=compression)

    # clean up column names
    clean_names = {k:k.strip().replace('.','_') for k in input_data.columns}
    input_data = input_data.rename(columns=clean_names)

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1)
    if not use_dataframe:
        X = X.values
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, feature_names

def get_sym_model(dataset, return_str=True):
    """return sympy model from dataset metadata"""
    metadata = load(
            open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
            Loader=Loader
    )
    df = pd.read_csv(dataset,sep='\t')
    features = [c for c in df.columns if c != 'target']
    description = metadata['description'].split('\n')
    model_str = [ms for ms in description if '=' in ms][0].split('=')[-1]
    if return_str:
        return model_str

    local_dict = {k: Symbol(f'x{i}') for i, k in enumerate(features)}
    local_dict['pi'] = sp.pi
    model_sym = parse_expr(model_str,
                           local_dict = local_dict,
                           evaluate=True)
    model_sym = round_floats(model_sym)
    return model_sym

def complexity(expr):
    c=0
    for arg in preorder_traversal(expr):
        c += 1
    return c

def round_floats(ex1):
    ex2 = ex1

    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a,Integer(0))
            else:
                ex2 = ex2.subs(a, Float(round(a, 3),3))
    return ex2

def rewrite_AIFeynman_model_size(model_str):
    """AIFeynman complexity was incorrect prior to version , update it here"""
    return complexity(parse_expr(model_str))


if __name__ == '__main__':
    print(token_to_index)
    # latex_expr = r'y=\cos{\left(2x_{1}^{2}xright)'
    latex_expr = r'2x_{1}'
    tokenized = tokenize_latex_to_char(latex_expr)
    print(tokenized)
