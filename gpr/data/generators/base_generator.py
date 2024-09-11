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
    def __init__(self, rng=None, verbose=False, *args, **kwargs):
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

    def __call__(self, num_variables: int = 5,
                 max_terms: int = 3,
                 num_realizations: int = 10,
                 real_numbers_realizations: bool = True,
                 allowed_operations: list = None,
                 keep_graph: bool = True,
                 keep_data: bool = False,
                 kmax: int = 5,
                 max_powers: int = 3,
                 real_const_decimal_places: int = 0,
                 constants_mean: float = 0.,
                 constants_var: float = 1.,
                 real_constants_min: float = -3.,
                 real_constants_max: float = 3.,
                 unary_operation_probability: float = 0.5,
                 nesting_probability: float = 0.5,
                 exponent_probability: float = 0.1,
                 constant_probability: float = 0.1,
                 max_depth: int = 2,
                 use_epsilon: bool = True,
                 max_const_exponent: int = 2,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls all method that lead to a realization."""
        # Check if keep_graph == False and keep_data == True. If we generate a
        # new graph we need to generate a new numpy array corresponding to that
        # graph.
        assert keep_graph or (not keep_data), "Cannot reuse the same data if the graph changes."

        max_try = 10
        for i in range(max_try):
            # If the keep_graph == False, generate a new graph. Do this when
            # self.graph is None too.
            # if (not keep_graph) or (self.graph is None):

            num_variables = self.rng.integers(1, num_variables + 1)
            # p = np.asarray([i for i in range(1, num_variables+1)])[::-1]
            # num_variables = self.rng.choice([i for i in range(1, num_variables+1)], p=p / sum(p))

            if (not keep_graph) or (self.graph is None):
                self.generate_random_graph(num_variables)
            # If the keep_data == False, generate a new numpy array with shape
            # (num_realizations, len(self.variables)). Here len(self.variables) ==
            # num_nodes. Do this when self.x_data is None too.
            # if (not keep_data) or (self.x_data is None):


            # generate an equation with max_terms < num_nodes and evaluate the
            # equation on the generated data.
            self.generate_equation(max_terms=max_terms,
                                allowed_operations=allowed_operations,
                                max_powers=max_powers,
                                real_const_decimal_places=real_const_decimal_places,
                                constants_mean=constants_mean,
                                constants_var=constants_var,
                                real_constants_min=real_constants_min,
                                real_constants_max=real_constants_max,
                                unary_operation_probability=unary_operation_probability,
                                nesting_probability=nesting_probability,
                                exponent_probability=exponent_probability,
                                constant_probability=constant_probability,
                                max_depth=max_depth,
                                use_epsilon=use_epsilon,
                                max_const_exponent=max_const_exponent,
                                **kwargs)

            realizations_tolerance = int(num_realizations * 0.1)
            num_realizations_plus = num_realizations + realizations_tolerance

            if (not keep_data) or (self.x_data is None):
                self.generate_data(num_realizations=num_realizations_plus,
                                   real_numbers_realizations=real_numbers_realizations,
                                   kmax=kmax)


            self.limit_constants(max_constant=real_constants_max, min_constant=real_constants_min)

            if self.expression == False or not self.contains_variables(self.expression, self.variables):
                continue

            x, y = self.evaluate_equation()

            nan_sum_y = np.sum(np.isnan(y) | np.isinf(y))

            nan_sum_x = np.sum(np.isnan(x) | np.isinf(x))

            if nan_sum_y > realizations_tolerance or nan_sum_x > 0:
                print("NaN or Inf values in y or x.")
                continue

            if  np.sum(np.isnan(y)) > 0:
                x = x[~np.isnan(y), :]
                y = y[~np.isnan(y)]

            if  np.sum(np.isinf(y)) > 0:
                x = x[~np.isinf(y), :]
                y = y[~np.isinf(y)]

            x, y = x[:num_realizations, :], y[:num_realizations]
            

            is_nan = torch.tensor([np.isnan(y).sum() != 0 or np.isinf(y).sum() != 0])

            m, e = BaseGenerator.get_mantissa_exp(x, y)

            if len(self.expression_latex) > 800: # TODO make this a parameter / check if it's necessary
                continue

            return m, e, self.expression, is_nan
        return None

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
                         use_math_constants: bool=False,
                         max_powers: int = 2,
                         real_const_decimal_places: int = 0,
                         constants_mean: float = 0.,
                         constants_var: float = 1.,
                         real_constants_min: float = -3.,
                         real_constants_max: float = 3.,
                         unary_operation_probability: float = 0.5,
                         nesting_probability: float = 0.5,
                         exponent_probability: float = 0.1,
                         constant_probability: float = 0.1,
                         max_depth: int = 2,
                         use_epsilon: bool = True,
                         max_const_exponent: int = 2,
                         epsilon: float = 1e-10,
                         kmax: int = 5,
                         **kwargs) -> None:
        """Generates an equation that will be applied as a functional mechanism."""
        if allowed_operations is None:
            allowed_operations = ["+", "-", "*", "/", "sin", "cos", "log", "exp", "**"]

        symbols = {var: sp.symbols(var) for var in self.variables}

        self.expression = self._generate_random_expression(symbols,
                                                        self.allowed_operations if hasattr(self, 'allowed_operations') else allowed_operations,
                                                        max_terms,
                                                        use_math_constants=use_math_constants,
                                                        max_powers=max_powers,
                                                        real_const_decimal_places=real_const_decimal_places,
                                                        constants_mean=constants_mean,
                                                        constants_var=constants_var,
                                                        real_constants_min=real_constants_min,
                                                        real_constants_max=real_constants_max,
                                                        unary_operation_probability=unary_operation_probability,
                                                        nesting_probability=nesting_probability,
                                                        exponent_probability=exponent_probability,
                                                        constant_probability=constant_probability,
                                                        max_depth=max_depth,
                                                        use_epsilon=use_epsilon,
                                                        max_const_exponent=max_const_exponent,
                                                        epsilon=epsilon,
                                                        kmax=kmax,
                                                        **kwargs)
        self.expression_str = str(self.expression.rhs)
        self.expression_latex = sp.latex(self.expression)

        #self.used_symbols = {str(var): var for var in self.expression.rhs.free_symbols}
        self.used_symbols = sorted(
            [str(var) for var in self.expression.rhs.free_symbols], key=lambda x: int(x[1:])
        )
        self.equation = sp.lambdify([symbols[var] for var in self.used_symbols],
                                    self.expression.rhs, docstring_limit=1000, # TODO make this a parameter
                                    modules="numpy")

    def sample_from_mixture(self, num_samples, num_dimensions, mean=0., var=1., kmax=5):
        # 1. Sample number of clusters and weights
        k = min(self.rng.integers(1, kmax + 1), num_samples)
        weights = self.rng.dirichlet(np.ones(k))

        # 2. Sample cluster parameters
        centroids = self.rng.normal(mean, var, (k, num_dimensions))
        scales = self.rng.uniform(0, var, (k, num_dimensions))
        distributions = self.rng.choice([self.rng.normal, self.rng.uniform], k)

        # 3. Generate data for each cluster
        data = np.zeros((num_samples, num_dimensions), dtype=np.float32)
        samples_per_cluster = np.round(weights * num_samples).astype(int)
        samples_per_cluster[-1] = num_samples - samples_per_cluster[:-1].sum()  # Ensures total is correct

        # Check if there are any negative or zero values in samples_per_cluster
        if np.any(samples_per_cluster <= 0):
            # If so, switch to one sample per distribution
            samples_per_cluster = np.ones(k, dtype=int)
            samples_per_cluster[:num_samples] = 1
            samples_per_cluster[num_samples:] = 0

        start = 0
        for i in range(k):
            end = start + samples_per_cluster[i]

            if distributions[i] == self.rng.normal:
                cluster_data = distributions[i](centroids[i], scales[i], (samples_per_cluster[i], num_dimensions))
            else:
                low = centroids[i] - scales[i]
                high = centroids[i] + scales[i]
                cluster_data = distributions[i](low, high, (samples_per_cluster[i], num_dimensions))
                
            if num_dimensions > 1:
                # Apply random rotation from Haar distribution
                rotation_matrix = self.random_orthogonal_matrix(num_dimensions)
                cluster_data = np.dot(cluster_data, rotation_matrix)

            data[start:end] = cluster_data
            start = end

        # Shuffle the data
        self.rng.shuffle(data)

        return data

    def generate_data(self, num_realizations: int, real_numbers_realizations: bool=True, kmax: int=5) -> None:
        if not self.graph:
            raise ValueError("Graph not initialized. Call generate_random_graph first.")

        num_variables = len(self.variables)

        x_data = self.sample_from_mixture(num_realizations, num_variables, kmax)

        if not real_numbers_realizations:
            x_data = np.round(x_data).astype(np.int32)
        
        self.x_data = x_data

    def random_orthogonal_matrix(self, n):
        """Generate a random orthogonal matrix from the Haar distribution."""
        return special_ortho_group.rvs(n)

    @staticmethod
    def clip_x_data(x_data, operation_ranges, expression):
        clipped_x_data = x_data.copy()
        
        limited_ranges = {
            (-1, 1): ["asin", "arcsin", "acos", "arccos", "atanh", "arctanh"],
            (0, np.inf): ["sqrt", "log", "ln"],
            (1, np.inf): ["acosh", "arccosh"]
        }
        
        for range_, ops in limited_ranges.items():
            if any(op in str(expression) for op in ops):
                for i in range(clipped_x_data.shape[1]):
                    clipped_x_data[:, i] = np.clip(clipped_x_data[:, i], range_[0], range_[1])
        
        return clipped_x_data

    def evaluate_equation(self) -> tuple[np.ndarray, np.ndarray]:
        """ This method indexes the currently generated data based on the used
        variables in the sampled equation. This way we can reuse the same data
        for multiple generated equations from the same graph, hence increasing
        efficiency."""
        if not self.equation:
            raise ValueError("Equation not initialized. Call generate_equation first.")

        # find indeces of used symbols
        idxs = np.where(np.isin(self.variables, self.used_symbols))[0]
        x_data_slice = self.x_data[:, idxs]
        
        clipped_x_data = self.clip_x_data(x_data_slice, self.operation_families, self.expression)
        self.x_data[:, idxs] = clipped_x_data
        
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

    def limit_constants(self, max_constant, min_constant):
        def replace_out_of_range_constants(x):
            if isinstance(x, sp.Number):
                if x > max_constant or x < min_constant:
                    new_value = self.rng.integers(min_constant, max_constant)
                    print(f"replacing {x} with {new_value}") if self.verbose else None
                    return new_value

            return x
        
        self.expression = self.expression.xreplace(
            {atom: replace_out_of_range_constants(atom) for atom in self.expression.atoms(sp.Number)})


    @staticmethod
    def contains_variables(expression, variables):
        if isinstance(expression, sp.Eq):
            expression = expression.rhs
        symbol_set = set(sp.Symbol(str(s)) for s in variables)
        expr_symbols = expression.free_symbols
        return bool(symbol_set.intersection(expr_symbols))
    
    @staticmethod
    def is_valid_expression(expression):
        if isinstance(expression, sp.Eq):
            expression = expression.rhs
    
        if expression == 0 or any(obj in expression.atoms() for obj in [sp.zoo, sp.oo, sp.nan, sp.S.ComplexInfinity]):
            return False
        return True

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

    @staticmethod
    def _contains_only_constants(expr):
        """
        Check if the expression contains only constants (numbers or mathematical constants).
        """
        # for atom in expr.atoms():
        #     print(f"atom: {atom}, is constant: {atom.is_constant()}")
        return all(atom.is_constant() for atom in expr.atoms())