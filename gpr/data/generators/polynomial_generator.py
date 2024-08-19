import numpy as np
import sympy as sp
from sympy import Mul, Add

from gpr.data.abstract import AbstractGenerator
from gpr.data.generators.base_generator import BaseGenerator

class PolynomialGenerator(BaseGenerator):

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations: list, max_terms: int, use_constants: bool=False, **kwargs) -> sp.Eq:
        """Generates a random polynomial involving the provided symbols."""
        valid_operations = {"+", "-", "*", "/", "log", "exp", "sin", "cos"}
        if not all(op in valid_operations for op in allowed_operations):
            raise ValueError(f"allowed_operations can only contain {valid_operations}")

        max_powers = kwargs.get('max_powers', 2)
        real_numbers_variables = kwargs.get('real_numbers_variables', False)
        allow_negative_coefficients = "-" in allowed_operations

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

                    if real_numbers_variables:
                        term = self.rng.uniform(-10, 10) if allow_negative_coefficients else self.rng.uniform(1, 10)
                    else:
                        term = self.rng.integers(-10, 10) if allow_negative_coefficients else self.rng.integers(1, 10)

                    for var in term_variables:
                        exponent = self.rng.integers(1, max_powers + 1)
                        term *= var ** exponent

                    has_x_term = True  # Set the flag to True as we have added a variable term

                operation = self.rng.choice(allowed_operations)
                if operation == "log":
                    term = sp.log(sp.Abs(term) + 1e-9) # ensure log argument is positive
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


if __name__ == '__main__':
    generator = PolynomialGenerator()
    mantissa, exponent, expression = generator(num_variables=3, num_realizations=100, max_terms=10,
                    max_powers=4, real_numbers_variables=False,
                                               allowed_operations=["+", "cos",
                                                                   "sin",
                                                                   "log",
                                                                   "exp"],
                                               keep_graph=False,
                                               keep_data=False)
    # print(mantissa)
    # print(exponent)
    print(expression)


