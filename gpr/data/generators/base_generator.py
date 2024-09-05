import networkx as nx
import numpy as np
import sympy as sp
import random
import torch
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

from gpr.data.abstract import AbstractGenerator

#TODO: make trees out of the DiGraph that represent the equations


class BaseGenerator(AbstractGenerator):
    def __init__(self, rng=None, verbose=False):
        super().__init__(rng=rng)

        self.rng = rng if rng is not None else np.random.default_rng()
        self.verbose = verbose

        self.operation_families = {
            "Arithmetic": ["+", "-", "*", "/"],
            "Exponential and Logarithmic": ["exp", "log", "ln"],
            "Trigonometric": ["sin", "cos", "tan", "cot"],
            "Inverse Trigonometric": ["asin", "acos", "atan", "acot"],
            "Hyperbolic": ["sinh", "cosh", "tanh", "coth"],
            "Inverse Hyperbolic": ["asinh", "acosh", "atanh", "acoth"],
            "Miscellaneous": ["sqrt", "abs", "sign", "^"]
        }

    def filter_families(self, allowed_operations):
        filtered_families = {}
        for family, ops in self.operation_families.items():
            filtered_ops = [op for op in ops if op in allowed_operations]
            if filtered_ops:
                filtered_families[family] = filtered_ops
        if self.verbose:
            print("Filtered families and their operations:")
            for family, ops in filtered_families.items():
                print(f"Family: {family}, Operations: {ops}")
        return filtered_families

    def sample_operation(self, filtered_families):
        if not filtered_families:
            raise ValueError("No allowed operations available for sampling.")
        family = random.choice(list(filtered_families.keys()))
        operation = random.choice(filtered_families[family])
        return operation

    def __call__(self, num_variables: int=5, 
                 max_terms: int=3,
                 num_realizations: int=10, 
                 real_numbers_realizations: bool=True,
                 allowed_operations: list=None, 
                 keep_graph: bool=True,
                 keep_data: bool=False, 
                 nan_threshold: float=0.1,
                 kmax: int=5,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls all method that lead to a realization."""
        # Check if keep_graph == False and keep_data == True. If we generate a
        # new graph we need to generate a new numpy array corresponding to that
        # graph.
        assert keep_graph or (not keep_data), "Cannot reuse the same data if the graph changes."

        while True:
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
                                kmax=kmax)

            # generate an equation with max_terms < num_nodes and evaluate the
            # equation on the generated data.

            self.generate_equation(max_terms=max_terms,
                                allowed_operations=allowed_operations,
                                **kwargs)


            x, y = self.evaluate_equation()
            skip, is_nan = self.check_nan_inf(y, nan_threshold)
            skip_large_const = self.check_large_constants(self.expression, max_constant=self.real_constants_max*2)
            if skip or skip_large_const:
                continue

            m, e = BaseGenerator.get_mantissa_exp(x, y)
            return m, e, self.expression, is_nan

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

    def generate_data(self, num_realizations: int, real_numbers_realizations: bool=True, kmax: int=5) -> None:
        """
        Generates num_realizations input values using a mixture of distributions (Uniform and Gaussian), centered around kmax random centroids with random rotations.

        Args:
            num_realizations (int): Number of data points to generate.
            real_numbers_realizations (bool): If True, use continuous distributions.
            kmax (int): Maximum number of clusters in the mixture.
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call generate_random_graph first.")

        num_variables = len(self.variables)
        
        # 1. Sample number of clusters and weights
        k = self.rng.integers(1, kmax + 1)
        weights = self.rng.dirichlet(np.ones(k))

        # 2. Sample cluster parameters
        centroids = self.rng.normal(0, 1, (k, num_variables))
        scales = self.rng.uniform(0.1, 2, (k, num_variables)) 
        distributions = self.rng.choice([self.rng.normal, self.rng.uniform], k)
        
        # 3. Generate data for each cluster
        x_data = np.zeros((num_realizations, num_variables), dtype=np.float32)
        samples_per_cluster = np.round(weights * num_realizations).astype(int)
        samples_per_cluster[-1] = num_realizations - samples_per_cluster[:-1].sum()  # Ensures total is correct
        
        start = 0
        for i in range(k):
            end = start + samples_per_cluster[i]
            if distributions[i] == self.rng.normal:
                cluster_data = distributions[i](centroids[i], scales[i], (samples_per_cluster[i], num_variables))
            # uniform distribution
            else:
                low = centroids[i] - scales[i]
                high = centroids[i] + scales[i]
                cluster_data = distributions[i](low, high, (samples_per_cluster[i], num_variables))
            
            # Apply random rotation from Haar distribution
            rotation_matrix = self.random_orthogonal_matrix(num_variables)
            cluster_data = np.dot(cluster_data, rotation_matrix)
            
            x_data[start:end] = cluster_data
            start = end
        
        # Shuffle the data
        self.rng.shuffle(x_data)

        if not real_numbers_realizations:
            x_data = np.round(x_data).astype(np.int32)
        
        # Scale the data to the desired interval
        # min_val, max_val = sample_interval
        # x_data = min_val + (max_val - min_val) * (x_data - x_data.min()) / (x_data.max() - x_data.min())
        
        self.x_data = x_data

    def random_orthogonal_matrix(self, n):
        """Generate a random orthogonal matrix from the Haar distribution."""
        return special_ortho_group.rvs(n)

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
        # TODO: this can throw a RunTimeWarning on overflow, can't be caught with except?
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

    def check_nan_inf(self, y: np.ndarray, nan_threshold: float) -> tuple[bool, torch.Tensor]:
        y = np.where(np.isinf(y), np.finfo(np.float16).max, y)
        nan_inf_count = np.isnan(y).sum() + np.isinf(y).sum()
        nan_inf_ratio = nan_inf_count / len(y)
        is_nan = torch.tensor([nan_inf_count > 0])
        
        if self.verbose:
            print(f"NaN/Inf ratio in y: {nan_inf_ratio}")
        
        if nan_inf_ratio > nan_threshold:
            if self.verbose:
                print("Skipping equation due to too many NaN/Inf values.")
            return True, is_nan
        
        return False, is_nan

    def check_large_constants(self, expression: sp.Eq, max_constant: float = 1e6) -> tuple[bool, torch.Tensor]:
        constants = list(expression.atoms(sp.Number))
        has_large_constant = any(abs(float(c)) > abs(max_constant) for c in constants)
        
        if self.verbose and has_large_constant:
            print(f"Skipping equation due to large constant(s): {[c for c in constants if abs(float(c)) > max_constant]}")
        
        return has_large_constant
