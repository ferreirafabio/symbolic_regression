import numpy as np
import sympy as sp
from sympy import Mul, Add
from sympy.core.function import _coeff_isneg

from gpr.data.abstract import AbstractGenerator
from gpr.data.generators.base_generator import BaseGenerator

class PolynomialGenerator(BaseGenerator):

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations: list, max_terms: int, use_constants: bool=False, **kwargs) -> sp.Eq:
        """
        Generates a random polynomial expression using the provided symbols and operations.

        Parameters:
        ----------
        symbols : dict, A dictionary of SymPy symbols representing variables.
        allowed_operations : list, List of operations to include: "+", "-", "*", "/", "log", "exp", "sin", "cos".
        max_terms : int, Maximum number of terms in the polynomial.
        use_constants : bool, optional, Whether to include constants like π and e (default is False).
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
        valid_operations = {"+", "-", "*", "/", "log", "exp", "sin", "cos"}
        if not all(op in valid_operations for op in allowed_operations):
            raise ValueError(f"allowed_operations can only contain {valid_operations}")

        max_powers = kwargs.get('max_powers', 2)
        allow_negative_coefficients = "-" in allowed_operations
        
        real_const_decimal_places = kwargs.get('real_const_decimal_places', 0)
        real_constants_min = kwargs.get('real_constants_min', 0.0)
        real_constants_max = kwargs.get('real_constants_max', 10.0)
        max_const_exponent = kwargs.get('max_const_exponent', 2)

        constants = [sp.pi, sp.E] if use_constants else []

        polynomial = sp.Integer(0)  # Initialize as zero to enter the loop

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

                    coefficient = self._generate_coefficient(
                        real_const_decimal_places, 
                        real_constants_min, 
                        real_constants_max, 
                        max_const_exponent, 
                        allow_negative_coefficients
                    )
                    term = coefficient

                    for var in term_variables:
                        exponent = self.rng.integers(1, max_powers + 1)
                        term *= var ** exponent

                    has_x_term = True  # Set the flag to True as we have added a variable term
                
                operation = self.rng.choice(allowed_operations)
                if operation == "log":
                    term = sp.log(sp.Abs(term) + 1) if real_const_decimal_places == 0 else sp.log(sp.Abs(term) + 1e-9)
                elif operation == "exp":
                    term = sp.exp(term % 3) # prevent exp from overflowing
                elif operation in ["sin", "cos"]:
                    term = getattr(sp, operation)(term)

                terms.append(term)

            # Ensure at least one term includes a variable
            if not has_x_term:
                continue

            polynomial = terms[0]

            for term in terms[1:]:
                operation = self.rng.choice(allowed_operations)
                if operation == "+":
                    polynomial = Add(polynomial, term, evaluate=True)
                elif operation == "-":
                    polynomial = Add(polynomial, -term, evaluate=True)
                elif operation == "*":
                    polynomial = Mul(polynomial, term, evaluate=True)
                elif operation == "/":
                    polynomial = Mul(polynomial, 1/term, evaluate=True)

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

        return polynomial
    
    def _generate_coefficient(self, decimal_places, min_val, max_val, max_exponent, allow_negative):
        """Generate a coefficient that could be either floating-point or integer."""
        if decimal_places > 0:
            base = self.rng.uniform(min_val, max_val)
            if allow_negative and self.rng.choice([True, False]):
                base = -base
            exponent = self.rng.integers(-max_exponent, max_exponent + 1)
            coefficient = f"{base:.{decimal_places}f}e{exponent:+}"
            return sp.Float(coefficient, decimal_places)
        else:
            coefficient = self.rng.integers(min_val, max_val)
            if allow_negative and self.rng.choice([True, False]):
                coefficient = -coefficient
            return sp.Integer(coefficient)



if __name__ == '__main__':
    generator = PolynomialGenerator()
    
    # Define the parameters for the equation generation
    params = {
        "num_variables": 3,
        "num_realizations": 1,  # We generate one realization per loop iteration
        "max_terms": 10,
        "max_powers": 4,
        "real_numbers_variables": False,
        "use_constants": True,
        "allowed_operations": ["+", "-", "cos", "sin", "log", "exp"],
        "keep_graph": False,
        "keep_data": False,
        "max_const_exponent": 3,
        "real_const_decimal_places": 3,
        "real_constants_min": -5,
        "real_constants_max": 5,
    }

    # Generate and print 5 different equations
    for i in range(5):
        mantissa, exponent, expression = generator(**params)
        print(f"Equation {i+1}: {expression}")


