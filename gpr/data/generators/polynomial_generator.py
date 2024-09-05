import sympy as sp
import random
import numpy as np
from gpr.data.abstract import AbstractGenerator
from gpr.data.generators.base_generator import BaseGenerator
from gpr.data.utils import format_floats_recursive


class PolynomialGenerator(BaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filtered_operator_families = None
        
        self.process_kwargs(**kwargs)

    def process_kwargs(self, **kwargs):
        self.max_powers = kwargs.get('max_powers', 2)
        self.real_const_decimal_places = kwargs.get('real_const_decimal_places', 0)
        self.real_constants_min = kwargs.get('real_constants_min', -3.)
        self.real_constants_max = kwargs.get('real_constants_max', 3.)
        self.unary_operation_probability = kwargs.get('unary_operation_probability', 0.5)
        self.nesting_probability = kwargs.get('nesting_probability', 0.5)
        self.exponent_probability = kwargs.get('exponent_probability', 0.1)
        self.epsilon = 1e-10 if self.real_const_decimal_places > 0 else 1

        print("Processed kwargs:")
        for key, value in self.__dict__.items():
            if key.startswith('_') or key in ['x_data', 'y_data']:
                continue
            if hasattr(value, 'shape') and len(value.shape) < 10:
                print(f"{key}: {value}")
            elif not hasattr(value, 'shape'):
                print(f"{key}: {value}")


    def _generate_term(self, symbols, allowed_operations, use_math_constants, depth, max_depth):
        constants = [sp.pi, sp.E] if use_math_constants else []
        use_constant = self.rng.choice([True, False]) if constants else False
        
        if use_constant:
            term = self.rng.choice(constants)
        else:
            num_vars_in_term = self.rng.integers(1, len(self.variables) + 1)
            term_variables = list(symbols.values())[:num_vars_in_term]

            if self.real_const_decimal_places > 0:
                term = round(
                    self.rng.uniform(self.real_constants_min, self.real_constants_max),
                    self.real_const_decimal_places
                )
            else:
                term = self.rng.integers(self.real_constants_min, self.real_constants_max)
            
            if "-" in allowed_operations:
                term *= self.rng.choice([-1, 1])
            
            for var in term_variables:
                if self.rng.random() < self.exponent_probability:
                    p = np.asarray([i for i in range(2, self.max_powers + 1)])[::-1]
                    exponent = self.rng.choice([i for i in range(2, self.max_powers + 1)], p=p/sum(p))
                    # print(f"exponent: {exponent} (p: {p})")
                    term *= var ** exponent
                else:
                    term *= var
        
        if self.rng.random() < self.unary_operation_probability:
            operation = self.sample_operation(self.filtered_operator_families)
            term = self.apply_unary_operation(term, operation)
        
        if depth < max_depth and self.rng.random() < self.nesting_probability:
            # print(f"nesting depth {depth}")
            nested_term = self._generate_term(symbols, allowed_operations, use_math_constants, depth + 1, max_depth)
            term = self.compose_terms(term, nested_term)
            term = format_floats_recursive(term, self.real_const_decimal_places)
        
        return term

    def _connect_terms(self, terms, allowed_operations):
        if not terms:
            return sp.Integer(0)
        
        polynomial = terms[0] 
        arithmetic_operations = self.operation_families["Arithmetic"]
        allowed_arithmetic = [op for op in arithmetic_operations if op in allowed_operations]

        for term in terms[1:]:
            operation = random.choice(allowed_arithmetic)
            polynomial = self.apply_arithmetic_operation(polynomial, term, operation)
        
        return polynomial

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations: list, max_terms: int, use_math_constants: bool=False, depth: int=0, max_depth: int=2, **kwargs) -> sp.Eq:
        """
        Generates a random polynomial expression using the provided symbols and operations.

        Parameters:
        ----------
        symbols : dict, A dictionary of SymPy symbols representing variables.
        allowed_operations : list, List of operations to include: "+", "-", "*", "/", "log", "exp", "sin", "cos".
        max_terms : int, Maximum number of terms in the polynomial.
        use_math_constants : bool, optional, Whether to include mathematical constants like pi and e (default is False).
        depth : int, Current depth of nested terms.
        max_depth : int, Maximum depth of nested terms.
        **kwargs : dict, Additional options, e.g., max_powers (int) for variable exponents,
            real_const_decimal_places (int) for decimal precision of real constants,
            real_constants_min (float) and real_constants_max (float) for setting the range of real constants.
            max_const_exponent (int) to set the maximum exponent value in scientific notation.

        Returns:
        -------
        sp.Eq, A SymPy equation representing the generated polynomial.

        Raises:
        ------
        ValueError, If any operation in `allowed_operations` is unsupported.
        """
        

        valid_operations = {"+", "-", "*", "/", '^', 'log', 'ln', 'exp', "sin", "cos", "tan", "cot", "asin", "acos", "atan", "acot", "sinh", "cosh", "tanh", "coth", "asinh", "acosh", "atanh", "acoth", "sqrt", "abs", "sign"}
        if not all(op in valid_operations for op in allowed_operations):
            raise ValueError(f"allowed_operations can only contain {valid_operations} but you supplied {allowed_operations} where type is {type(allowed_operations)}")

        if self.filtered_operator_families is None:
            self.filtered_operator_families = self.filter_families(allowed_operations)

        while True:
            num_terms = self.rng.integers(1, max_terms + 1)
            terms = []
            has_x_term = False

            for _ in range(num_terms):  
                term = self._generate_term(symbols, allowed_operations, use_math_constants, depth, max_depth)
                terms.append(term)
                if any(sym in term.free_symbols for sym in symbols.values()):
                    has_x_term = True

            if not has_x_term:
                continue

            polynomial = self._connect_terms(terms, allowed_operations)
            polynomial = format_floats_recursive(polynomial, self.real_const_decimal_places)
            polynomial = sp.simplify(polynomial)

            if not polynomial.has(*symbols.values()):
                continue

            if polynomial == 0 or any(obj in polynomial.atoms() for obj in [sp.zoo, sp.oo, sp.nan, sp.S.ComplexInfinity]):
                continue

            eq = sp.Eq(polynomial, 0)
            if isinstance(eq, sp.Equality):
                polynomial = self.canonicalize_equation(eq)

            if polynomial != 0 and polynomial != sp.S.false and polynomial != sp.S.true:
                break

            polynomial = format_floats_recursive(polynomial, self.real_const_decimal_places)

        return polynomial
    
    def apply_unary_operation(self, term, operation):
        if operation == "log":
            return sp.log(sp.Abs(term + self.epsilon), 10)
        elif operation == "ln":
            return sp.ln(sp.Abs(term + self.epsilon))
        elif operation == "exp":
            return sp.exp(term)
        elif operation in ["asin", "acos", "atan", "acot", "asinh", "acosh", "atanh", "acoth", "sin", "cos", "tan", "cot", "sinh", "cosh", "tanh", "coth"]:
            return getattr(sp, operation)(term)
        elif operation == "sqrt":
            return sp.sqrt(sp.Abs(term))
        elif operation == "abs":
            return sp.Abs(term)
        elif operation == "sign":
            return sp.sign(term)
        else:
            return term

    def apply_arithmetic_operation(self, term1, term2, operation):
        if operation == "+":
            return sp.Add(term1, term2, evaluate=True)
        elif operation == "-":
            return sp.Add(term1, -term2, evaluate=True)
        elif operation == "*":
            return sp.Mul(term1, term2, evaluate=True)
        elif operation == "/":
            return sp.Mul(term1, 1/term2, evaluate=True)
        else:
            raise ValueError(f"Unsupported arithmetic operation: {operation}")

    def compose_terms(self, outer_term, inner_term):
        # print(f"composing outer term: {outer_term} with inner term: {inner_term} with result: {outer_term.subs(self.variables[0], inner_term)}")
        return outer_term.subs(self.variables[0], inner_term)

if __name__ == '__main__':
    from gpr.data.utils import tokenize_latex_to_char, token_to_index
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    def detokenize_indices_to_latex(indices):
        """
        Convert a list of token indices back to a LaTeX string.
        """
        return ''.join(index_to_token[idx] for idx in indices)

    generator = PolynomialGenerator(verbose=True)
    
    # Define the parameters for the equation generation
    params = {
        "num_variables": 6,
        "num_realizations": 256,  # We generate one realization per loop iteration
        "max_terms": 4,
        "max_powers": 3,
        "use_constants": True,
        # "allowed_operations": ["+", "-", "*", "/", "exp", "cos", "sin", "log", "ln", "sqrt"],
        # "allowed_operations": ["+", "-", "*", "/", 'log', 'ln', 'exp', "sin", "cos", "tan", "cot","cosh","tanh","coth", 'sqrt', 'abs', 'sign'],
        # "allowed_operations": ["+", "-", "*", "/", "exp", "sqrt", "log", 'ln', 'exp', "asin", "acos", "atan", "acot", "asinh", "acosh", "atanh", "acoth", "sin", "cos", "tan", "cot", "sinh", "cosh", "tanh", "coth", "abs", "sign"],
        # "allowed_operations": ["+", "-", "*", "/", "log", "sin", "cos", "tan", "exp"],
        "allowed_operations": ["+", "-", "*", "/", "sin", "cos", "asin", "acos", "sqrt"],
        "keep_graph": False,
        "keep_data": False,
        "use_epsilon": True,
        "max_const_exponent": 2,
        "real_const_decimal_places": 0,
        "real_constants_min": -2.,
        "real_constants_max": 2.,
        "nan_threshold": 0.5,
        "max_depth": 4, # 0-indexed depth
        "nesting_probability": 0.8,
        "unary_operation_probability": 0.2,
        "kmax": 5,
        "exponent_probability": 0.1,
    }

    # Generate and print 5 different equations
    for i in range(50):
        mantissa, exponent, expression, is_nan = generator(**params)
        print(f"Equation {i+1}: {expression} expression contains NaNs: {is_nan}")
        
        # def get_constants(equation):
        #     return list(equation.atoms(sp.Number))
    
        # constants = get_constants(expression)
        # print(f"Constants: {constants}")

        # latex_string = sp.latex(expression)
        # print(f"Equation {i+1}: {latex_string}")

        # tokenized = tokenize_latex_to_char(latex_string)
        # print(f"Tokenized: {tokenized}")

        # Detokenize back to LaTeX string
        # detokenized_latex = detokenize_indices_to_latex(tokenized)
        # print(f"Detokenized back to LaTeX: {detokenized_latex}")
