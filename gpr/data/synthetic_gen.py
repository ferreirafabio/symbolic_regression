import networkx as nx
import numpy as np
import sympy as sp
import random
import matplotlib.pyplot as plt

#TODO: convert to format as specified by the implementation from Fabio
#TODO: add it to the DataLoader
#TODO: make trees out of the DiGraph that represent the equations

class DataGenerator:
    def __init__(self):
        self.graph = None
        self.equation = None
        self.variables = []
        self.equation_str = ""

    def generate_random_graph(self, num_nodes=5, num_edges=10):
        """Generates a complex hierarchical random graph."""
        self.graph = nx.DiGraph()

        # Add nodes representing variables
        for i in range(1, num_nodes + 1):
            self.graph.add_node(f"x{i}")

        # Add edges representing relationships between variables
        while len(self.graph.edges) < num_edges:
            u, v = np.random.choice(list(self.graph.nodes), 2, replace=False)
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)

        # Ensure the graph remains acyclic
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph = nx.DiGraph([(f"x{i}", f"x{i+1}") for i in range(1, num_nodes)])

        self.variables = sorted(self.graph.nodes)

    def _generate_random_expression(self, symbols, allowed_operations, max_terms):
        """Generates a random mathematical expression involving the provided symbols."""
        expression = 0
        num_terms = min(max_terms, len(symbols))
        selected_vars = random.sample(list(symbols.keys()), num_terms)

        used_operations = []

        for var in selected_vars:
            operation = random.choice(allowed_operations)

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
                coeff = np.random.uniform(-5, 5)
                if operation == "/":
                    expression += symbols[var] / (coeff if coeff != 0 else 1)  # avoid division by zero
                else:
                    expr_op = {'+': sp.Add, '-': sp.Mul, '*': sp.Mul, '/': sp.div}[operation]
                    expression = expr_op(expression, coeff * symbols[var])

        return expression

    def generate_equation(self, allowed_operations=None, max_terms=5):
        """Generates an equation that will be applied as a functional mechanism."""
        if allowed_operations is None:
            allowed_operations = ["+", "-", "*", "/", "sin", "cos", "log", "exp", "**"]

        symbols = {var: sp.symbols(var) for var in self.variables}
        expression = self._generate_random_expression(symbols, allowed_operations, max_terms)
        self.equation = sp.lambdify(list(symbols.values()), expression, modules="numpy")
        self.equation_str = str(expression)

    def generate_data(self, num_points=100):
        """Generates a dataset based on the random graph and the equation."""
        if not self.graph or not self.equation:
            raise ValueError("Graph or equation not initialized. Call generate_random_graph and generate_equation first.")

        x_data = np.zeros((num_points, len(self.variables)))

        # Randomly initialize values for all variables
        for i, var in enumerate(self.variables):
            x_data[:, i] = np.random.uniform(0, 10, num_points)

        # Apply the equation's functional mechanism
        y_data = self.equation(*[x_data[:, i] for i in range(len(self.variables))])

        return x_data, y_data

    def visualize_data(self, x_data, y_data):
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

if __name__ == '__main__':
    # Example usage:
    generator = DataGenerator()
    generator.generate_random_graph(num_nodes=20, num_edges=10)
    generator.generate_equation(allowed_operations=["+", "*", "sin", "cos",
                                                    "log"], max_terms=10)
    x, y = generator.generate_data(num_points=100)

    generator.visualize_data(x, y)

