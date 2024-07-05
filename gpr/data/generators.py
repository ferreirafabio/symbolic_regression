import networkx as nx
import numpy as np
import sympy as sp
import random
import torch
import matplotlib.pyplot as plt

from sympy import Mul, Add
from gpr.data.abstract import AbstractGenerator

#TODO: make trees out of the DiGraph that represent the equations


class BaseGenerator(AbstractGenerator):
    def __init__(self, rng=None):
        super().__init__(rng=rng)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, num_nodes: int=5, num_edges: int=5, max_terms: int=3,
                 num_realizations: int=10, real_numbers_realizations: bool=True,
                 allowed_operations: list=None, keep_graph: bool=True,
                 keep_data: bool=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls all method that lead to a realization."""

        # Check if keep_graph == False and keep_data == True. If we generate a
        # new graph we need to generate a new numpy array corresponding to that
        # graph.
        assert keep_graph or (not keep_data), "Cannot reuse the same data if the graph changes."

        # If the keep_graph == False, generate a new graph. Do this when
        # self.graph is None too.
        if (not keep_graph) or (self.graph is None):
            self.generate_random_graph(num_nodes, num_edges)
        # If the keep_data == False, generate a new numpy array with shape
        # (num_realizations, len(self.variables)). Here len(self.variables) ==
        # num_nodes. Do this when self.x_data is None too.
        if (not keep_data) or (self.x_data is None):
            self.generate_data(num_realizations=num_realizations)

        # generate an equation with max_terms < num_nodes and evaluate the
        # equation on the generated data.
        self.generate_equation(max_terms=max_terms, allowed_operations=allowed_operations)
        x, y = self.evaluate_equation()
        m, e = BaseGenerator.get_mantissa_exp(x, y)
        return m, e, self.expression

    def generate_random_graph(self, num_nodes: int, num_edges: int) -> None:
        """Generates a random graph."""
        self.graph = nx.DiGraph()
        if num_edges > (num_nodes * (num_nodes-1) / 2):
            num_edges = (num_nodes * (num_nodes-1))/2
        # Add nodes representing variables
        for i in range(1, num_nodes + 1):
            self.graph.add_node(f"x{i}")

        # Add edges representing relationships between variables
        while len(self.graph.edges) < num_edges:
            u, v = self.rng.choice(list(self.graph.nodes), 2, replace=False)
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)

        # Ensure the graph remains acyclic
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph = nx.DiGraph([(f"x{i}", f"x{i+1}") for i in range(1, num_nodes)])

        self.variables = sorted(self.graph.nodes, key=lambda x: int(x[1:]))

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations:
                                    list, max_terms: int, **kwargs) -> sp.Eq:
        return NotImplementedError('Need to add an expression generator')

    def generate_equation(self, max_terms: int, allowed_operations: list=None,
                         **kwargs) -> None:
        """Generates an equation that will be applied as a functional mechanism."""
        if allowed_operations is None:
            allowed_operations = ["+", "-", "*", "/", "sin", "cos", "log", "exp", "**"]

        symbols = {var: sp.symbols(var) for var in self.variables}
        self.expression = self._generate_random_expression(symbols,
                                                           self.allowed_operations if hasattr(self, 'allowed_operations') else allowed_operations,
                                                           max_terms, **kwargs)
        self.expression_str = str(self.expression.rhs)
        self.expression_latex = sp.latex(self.expression)

        #self.used_symbols = {str(var): var for var in self.expression.rhs.free_symbols}
        self.used_symbols = sorted(
            [str(var) for var in self.expression.rhs.free_symbols], key=lambda x: int(x[1:])
        )
        self.equation = sp.lambdify([symbols[var] for var in self.used_symbols],
                                    self.expression.rhs,
                                    modules="numpy")

    def generate_data(self, num_realizations: int, real_numbers_realizations: bool=True) -> None:
        """Generates a dataset based on the random graph."""
        if not self.graph:
            raise ValueError("Graph not initialized. Call generate_random_graph first.")

        dist = self.rng.uniform if real_numbers_realizations else self.rng.randint
        x_data = dist(-10, 10, size=(num_realizations,
                                     len(self.variables))).astype('float32')
        self.x_data = x_data

    def evaluate_equation(self) -> tuple[np.ndarray, np.ndarray]:
        """ This method indexes the currently generated data based on the used
        variables in the sampled equation. This way we can reuse the same data
        for multiple generated equations from the same graph, hence increasing
        efficiency."""
        if not self.equation:
            raise ValueError("Equation not initialized. Call generate_equation first.")

        # find indeces of used symbols
        idxs = np.where(np.isin(self.variables, self.used_symbols))[0]
        # Apply the equation's functional mechanism
        y_data = self.equation(*[self.x_data[:, i] for i in idxs])
        #y_data = self.equation(*[self.x_data[:, i] for i in range(len(self.variables))])
        self.y_data = y_data

        return self.x_data[:, idxs], y_data

    @staticmethod
    def get_mantissa_exp(x_data: np.ndarray, y_data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x_mant, x_exp = torch.tensor(x_data, dtype=torch.float32).frexp()
        y_mant, y_exp = torch.tensor(y_data, dtype=torch.float32).frexp()
        
        mantissa = torch.cat([y_mant.unsqueeze(1), x_mant], dim=1)
        exponent = torch.cat([y_exp.unsqueeze(1), x_exp], dim=1)

        return mantissa, exponent

    @staticmethod
    def visualize_data(x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Visualizes the relationships between the input variables and the output."""
        num_vars = x_data.shape[1]
        fig, axs = plt.subplots(1, num_vars, figsize=(5 * num_vars, 5))

        if num_vars == 1:
            axs = [axs]  # Ensure we can index into the list even for a single subplot

        for i, ax in enumerate(axs):
            ax.scatter(x_data[:, i], y_data, alpha=0.7)
            ax.set_xlabel(f"Variable x{i + 1}")
            ax.set_ylabel("Output y")
            ax.set_title(f"Variable x{i + 1} vs Output y")

        plt.tight_layout()
        plt.savefig('fig.pdf')
    

    @staticmethod
    def canonicalize_equation(eq: sp.Eq) -> sp.Eq:
        """
        takes a sympy equation and returns its canonical form.
        The canonical form has terms sorted and combined, ensuring that
        similar equations always map to the same representation.
        """
        # first simplify the equation, then sort the terms and combine them again
        lhs, rhs = eq.lhs, eq.rhs
        expanded_exp = sp.expand(lhs - rhs)
        simplified_exp = sp.simplify(expanded_exp)
        ordered_terms = simplified_exp.as_ordered_terms()
        return sum(ordered_terms)


class RandomGenerator(BaseGenerator):
    
    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations:
                                    list, max_terms: int, **kwargs) -> sp.Eq:
        """Generates a random mathematical expression involving the provided symbols."""
        expression = 0
        num_terms = min(max_terms, len(symbols))
        selected_vars = random.sample(list(symbols.keys()), num_terms)

        used_operations = []

        for var in selected_vars:
            operation = self.rng.choice(allowed_operations)

            # Apply constraints on the usage of certain operations
            if operation == "log" and "log" not in used_operations:
                expression += sp.log(symbols[var] + 1)  # adjust log for computational stability
                used_operations.append("log")
            elif operation == "exp" and used_operations.count("exp") < 2:  # limit the number of exp used
                expression += sp.exp(symbols[var] % 3)  # Modulus to keep the exponent small
                used_operations.append("exp")
            elif operation in ["sin", "cos"] and used_operations.count(operation) < 2:
                expression += getattr(sp, operation)(symbols[var])
                used_operations.append(operation)
            elif operation in ["+", "-", "*", "/"]:
                coeff = self.rng.uniform(-5, 5)
                if operation == "/":
                    expression += symbols[var] / (coeff if coeff != 0 else 1)  # avoid division by zero
                else:
                    expression = sp.sympify(
                        f"{str(expression)}{operation}{coeff}*{var}",
                        evaluate=False
                    )

        return expression


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
    # from gpr.utils.configuration import Config
    # generator = RandomGenerator()
    # generator.generate_random_graph(num_nodes=20, num_edges=10)
    # generator.generate_data(num_realizations=100)

    # generator.generate_equation(allowed_operations=["+", "*", "sin", "cos"], max_terms=10)
    # x, y = generator.evaluate_equation()
    # m, e = generator.get_mantissa_exp(x, y)

    # # use the same generated data for different equations
    # generator.generate_equation(max_terms=10, allowed_operations=["+", "*", "sin", "cos"])
    # x, y = generator.evaluate_equation()
    # m, e = generator.get_mantissa_exp(x, y)

    generator = PolynomialGenerator()
    mantissa, exponent, expression = generator(num_nodes=3, num_edges=3, num_realizations=100, max_terms=10,
                    max_powers=4, real_numbers_variables=False, allowed_operations=["+", "cos", "sin", "log", "exp"])
    # print(mantissa)
    # print(exponent)
    print(expression)


