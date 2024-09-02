import sympy as sp
import random

from gpr.data.abstract import AbstractGenerator
from gpr.data.generators.base_generator import BaseGenerator
from gpr.data.utils import format_floats_recursive


class PolynomialGenerator(BaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filtered_operator_families = None
        

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations: list, max_terms: int, use_math_constants: bool=False, **kwargs) -> sp.Eq:
        """
        Generates a random polynomial expression using the provided symbols and operations.

        Parameters:
        ----------
        symbols : dict, A dictionary of SymPy symbols representing variables.
        allowed_operations : list, List of operations to include: "+", "-", "*", "/", "log", "exp", "sin", "cos".
        max_terms : int, Maximum number of terms in the polynomial.
        use_math_constants : bool, optional, Whether to include mathematical constants like pi and e (default is False).
        **kwargs : dict, Additional options, e.g., max_powers (int) for variable exponents,
            real_const_decimal_places (int) for decimal precision of real constants,
            real_constants_min (float) and real_constants_max (float) for setting the range of real constants.
            max_const_exponent (int) to set the maximum exponent value in scientific notation.

        Returns:
        -------

        Returns:
        -------
        sp.Eq, A SymPy equation representing the generated polynomial.

        Raises:
        ------
        ValueError, If any operation in `allowed_operations` is unsupported.
        """
        valid_operations = {"+", "-", "*", "/", '^', 'log', 'ln', 'exp', "sin", "cos", "tan", "cot","asin","acos","atan","acot","sinh","cosh","tanh","coth","asinh","acosh","atanh","acoth", 'sqrt', 'abs', 'sign'}
        if not all(op in valid_operations for op in allowed_operations):
            raise ValueError(f"allowed_operations can only contain {valid_operations} but you supplied {allowed_operations} where type is {type(allowed_operations)}")

        if self.filtered_operator_families is None:
            self.filtered_operator_families = self.filter_families(allowed_operations)

        max_powers = kwargs.get('max_powers', 2)
        allow_negative_coefficients = "-" in allowed_operations
        
        real_const_decimal_places = kwargs.get('real_const_decimal_places', 0)
        real_constants_min = kwargs.get('real_constants_min', 0.0)
        real_constants_max = kwargs.get('real_constants_max', 10.0)
        max_const_exponent = kwargs.get('max_const_exponent', 2)

        constants = [sp.pi, sp.E] if use_math_constants else []

        polynomial = sp.Integer(0)  # Initialize as zero to enter the loop

        epsilon = 1e-10 if real_const_decimal_places > 0 else 1  # Small positive value to avoid log(0) and ln(0)

        while True:
            num_terms = self.rng.integers(1, max_terms + 1)
            terms = []
            has_x_term = False  # Flag to check if any term includes a variable

            for _ in range(num_terms):
                # Randomly decide whether to use a variable or a constant if constants are allowed
                use_constant = self.rng.choice([True, False]) if constants else False
                if use_constant and not has_x_term:
                    term = self.rng.choice(constants)
                else:
                    num_vars_in_term = self.rng.integers(1, len(self.variables) + 1)
                    term_variables = list(symbols.values())[:num_vars_in_term]

                    # case: floating point constants
                    if real_const_decimal_places > 0:
                        term = round(
                            self.rng.uniform(real_constants_min, real_constants_max),
                            real_const_decimal_places
                        )
                    # case: integer constants
                    else:
                        term = self.rng.integers(real_constants_min, real_constants_max)

                    if allow_negative_coefficients:
                        term *= self.rng.choice([-1, 1])

                    for var in term_variables:
                        exponent = self.rng.integers(1, max_powers + 1)
                        term *= var ** exponent

                    has_x_term = True  # Set the flag to True as we have added a variable term

                # Main logic
                operation = self.sample_operation(self.filtered_operator_families)

                if operation == "log":
                    term = sp.log(sp.Abs(term + epsilon), 10)
                elif operation == "ln":
                    term = sp.ln(sp.Abs(term + epsilon))
                elif operation == "exp":                    
                    term = sp.exp(term)
                elif operation in ["asin", "acos", "atan", "acot", "asinh", "acosh", "atanh", "acoth", "sin", "cos", "tan", "cot", "sinh", "cosh", "tanh", "coth"]:
                    term = getattr(sp, operation)(term)
                elif operation == "sqrt":
                    term = sp.sqrt(sp.Abs(term))
                elif operation == "abs":
                    term = sp.Abs(term)
                elif operation == "sign":
                    term = sp.sign(term)

                terms.append(term)

                # Ensure at least one term includes a variable
                if not has_x_term:
                    continue

                polynomial = terms[0]

                for term in terms[1:]:
                    # operation = self.rng.choice(allowed_operations)
                    operation = self.sample_operation(self.filtered_operator_families)
                    if operation == "+":
                        polynomial = sp.Add(polynomial, term, evaluate=True)
                    elif operation == "-":
                        polynomial = sp.Add(polynomial, -term, evaluate=True)
                    elif operation == "*":
                        polynomial = sp.Mul(polynomial, term, evaluate=True)
                    elif operation == "/":
                        polynomial = sp.Mul(polynomial, 1/term, evaluate=True)
                    elif operation == "^":
                        polynomial = sp.Pow(polynomial, term)

            polynomial = format_floats_recursive(polynomial, real_const_decimal_places)

            polynomial = sp.simplify(polynomial)  # Simplify to combine like terms

            # Ensure the polynomial includes at least one variable term
            if not polynomial.has(*symbols.values()):
                continue

            # Check for invalid objects in the polynomial
            if polynomial == 0 or any(obj in polynomial.atoms() for obj in [sp.zoo, sp.oo, sp.nan, sp.S.ComplexInfinity]):
                continue
        
            # TODO: check why some polynomials get transformed into a BooleanFalse instead of sp.Equality when cast into sp.Eq
            eq = sp.Eq(polynomial, 0)
            if isinstance(polynomial, sp.Equality):
                polynomial = self.canonicalize_equation(eq)

            if polynomial != 0 and polynomial != sp.S.false and polynomial != sp.S.true:
                break

            # Post-process the polynomial to enforce decimal places recursively
            polynomial = format_floats_recursive(polynomial, real_const_decimal_places)


        return polynomial
    


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
        "num_variables": 3,
        "num_realizations": 10000,  # We generate one realization per loop iteration
        "max_terms": 6,
        "max_powers": 2,
        "use_constants": True,
        "allowed_operations": ["+", "-", "*", "/", "exp", "cos", "sin", "log", "ln", "sqrt"],
        # "allowed_operations": ["+", "-", "*", "/", 'log', 'ln', 'exp', "sin", "cos", "tan", "cot","cosh","tanh","coth", 'sqrt', 'abs', 'sign'],
        "keep_graph": False,
        "keep_data": False,
        "use_epsilon": True,
        "max_const_exponent": 2,
        "real_const_decimal_places": 2,
        "real_constants_min": -5,
        "real_constants_max": 5,
        "nan_inf_threshold": 0.5,
    }

    # Generate and print 5 different equations
    for i in range(50):
        mantissa, exponent, expression, valid = generator(**params)
        print(f"Equation {i+1}: {expression} expression contains NaNs/Infs: {not valid}")
        # latex_string = sp.latex(expression)
        # print(f"Equation {i+1}: {latex_string}")

        # tokenized = tokenize_latex_to_char(latex_string)
        # print(f"Tokenized: {tokenized}")

        # Detokenize back to LaTeX string
        # detokenized_latex = detokenize_indices_to_latex(tokenized)
        # print(f"Detokenized back to LaTeX: {detokenized_latex}")