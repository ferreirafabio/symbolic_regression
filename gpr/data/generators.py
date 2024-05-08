import networkx as nx
import numpy as np
import sympy as sp
import random
import torch
import matplotlib.pyplot as plt

from gpr.data.abstract import AbstractGenerator

#TODO: add it to the DataLoader
#TODO: make trees out of the DiGraph that represent the equations


class SimpleGenerator(AbstractGenerator):
    def __init__(self, rng=None):
        super().__init__(rng=rng)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, num_nodes: int, num_edges: int, max_terms: int,
                 num_realizations: int, real_numbers_realizations: bool=True,
                 allowed_operations: list=None, keep_graph: bool=True,
                 keep_data: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls all method that lead to a realization."""

        if (not keep_graph) or (self.graph is None) or (self.x_data is None):
            self.generate_random_graph(num_nodes, num_edges)
            self.generate_data(num_points=num_realizations)
        if (not keep_data) or (self.x_data is None):
            self.generate_data(num_points=num_realizations)

        self.generate_equation(max_terms=max_terms, allowed_operations=allowed_operations)
        x, y = self.evaluate_equation()
        m, e = SimpleGenerator.get_mantissa_exp(x, y)
        return m, e

    def generate_random_graph(self, num_nodes: int, num_edges: int) -> None:
        """Generates a complex hierarchical random graph."""
        self.graph = nx.DiGraph()

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

    def _generate_random_expression(self, symbols: dict, allowed_operations: list, max_terms: int) -> sp.Eq:
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
                    expr_op = {'+': sp.Add, '-': sp.Mul, '*': sp.Mul, '/': sp.div}[operation]
                    expression = expr_op(expression, coeff * symbols[var])

        y = sp.symbols('y')
        equation = sp.Eq(y, expression)
        return equation

    def generate_equation(self, max_terms: int, allowed_operations: list=None) -> None:
        """Generates an equation that will be applied as a functional mechanism."""
        if allowed_operations is None:
            allowed_operations = ["+", "-", "*", "/", "sin", "cos", "log", "exp", "**"]

        symbols = {var: sp.symbols(var) for var in self.variables}
        self.expression = self._generate_random_expression(symbols, allowed_operations, max_terms)
        self.expression_str = str(self.expression.rhs)
        self.expression_latex = sp.latex(self.expression)

        #self.used_symbols = {str(var): var for var in self.expression.rhs.free_symbols}
        self.used_symbols = sorted(
            [str(var) for var in self.expression.rhs.free_symbols], key=lambda x: int(x[1:])
        )
        self.equation = sp.lambdify([symbols[var] for var in self.used_symbols],
                                    self.expression.rhs,
                                    modules="numpy")

    def generate_data(self, num_points: int, real_numbers_realizations: bool=True) -> None:
        """Generates a dataset based on the random graph."""
        if not self.graph:
            raise ValueError("Graph not initialized. Call generate_random_graph first.")

        dist = self.rng.uniform if real_numbers_realizations else self.rng.randint
        x_data = dist(-10, 10, size=(num_points,
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


#class PolynomialGenerator(SimpleGenerator):
    #def _generate_random_expression(self, symbols: dict, allowed_operations: list, max_terms: int) -> sp.Eq:
        #"""Generates a random polynomial involving the provided symbols."""
        #expression = 0
        #num_terms = self.rng.integers(1, max_terms + 1)
#
        #for _ in range(num_terms):
            #num_vars_in_term = rng.integers(1, len(self.variables) + 1)  # exclusive upper bound
            #term_variables = list(symbols.values())[:num_vars_in_term]
            #term = rng.uniform(-10, 10)
#
            #for var in term_variables:
                #exponent = self.rng.integers(1, max_powers + 1)
                #term *= var ** exponent
            #expression += term
#
        #y = sp.symbols('y')
        #equation = sp.Eq(y, expression)
        #return equation


if __name__ == '__main__':
    generator = SimpleGenerator()
    generator.generate_random_graph(num_nodes=20, num_edges=10)
    generator.generate_data(num_points=100)

    generator.generate_equation(allowed_operations=["+", "*", "sin", "cos"], max_terms=10)
    x, y = generator.evaluate_equation()
    m, e = generator.get_mantissa_exp(x, y)

    # use the same generated data for different equations
    generator.generate_equation(max_terms=10, allowed_operations=["+", "*", "sin", "cos"])
    x, y = generator.evaluate_equation()
    m, e = generator.get_mantissa_exp(x, y)

    SimpleGenerator.visualize_data(x, y)


