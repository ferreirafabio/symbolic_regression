import networkx as nx
import numpy as np
import sympy as sp
import random
import torch
import matplotlib.pyplot as plt

from gpr.data.abstract import AbstractGenerator

#TODO: make trees out of the DiGraph that represent the equations


class BaseGenerator(AbstractGenerator):
    def __init__(self, rng=None):
        super().__init__(rng=rng)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, num_variables: int=5, max_terms: int=3,
                 num_realizations: int=10, real_numbers_realizations: bool=True,
                 allowed_operations: list=None, keep_graph: bool=True,
                 keep_data: bool=False, sample_interval: list=[-10, 10],
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls all method that lead to a realization."""

        # Check if keep_graph == False and keep_data == True. If we generate a
        # new graph we need to generate a new numpy array corresponding to that
        # graph.
        assert keep_graph or (not keep_data), "Cannot reuse the same data if the graph changes."

        # If the keep_graph == False, generate a new graph. Do this when
        # self.graph is None too.
        if (not keep_graph) or (self.graph is None):
            self.generate_random_graph(num_variables)
        # If the keep_data == False, generate a new numpy array with shape
        # (num_realizations, len(self.variables)). Here len(self.variables) ==
        # num_nodes. Do this when self.x_data is None too.
        if (not keep_data) or (self.x_data is None):
            self.generate_data(num_realizations=num_realizations,
                               real_numbers_realizations=real_numbers_realizations,
                               sample_interval=sample_interval)

        # generate an equation with max_terms < num_nodes and evaluate the
        # equation on the generated data.
        self.generate_equation(max_terms=max_terms,
                               allowed_operations=allowed_operations,
                               **kwargs)
        x, y = self.evaluate_equation()
        m, e = BaseGenerator.get_mantissa_exp(x, y)
        return m, e, self.expression

    def generate_random_graph(self, num_variables: int) -> None:
        """Generates a random graph."""
        num_nodes = num_variables # just a convention for now

        self.graph = nx.DiGraph()
        #if num_edges > (num_nodes * (num_nodes-1) / 2):
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

    def generate_data(self, num_realizations: int, real_numbers_realizations:
                      bool=True, sample_interval: list=[-10, 10]) -> None:
        """Generates a dataset based on the random graph."""
        if not self.graph:
            raise ValueError("Graph not initialized. Call generate_random_graph first.")

        dist = self.rng.uniform if real_numbers_realizations else self.rng.randint
        x_data = dist(sample_interval[0], sample_interval[1],
                      size=(num_realizations,
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


