import numpy as np
import sympy as sp
import torch
import functools
import inspect
from abc import ABCMeta, abstractmethod


class PrintMixin:
    def __repr__(self):
        cls = self.__class__.__name__
        idhex = hex(id(self))
        attrs = " ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        attrs = ": " + attrs if attrs else ""
        return f"<{cls} at {idhex}{attrs}>"


class AbstractSignatureChecker(ABCMeta):
    """
    Meta class for strictly enforcig signatures of @abstractmethod's in an abstract base class
    https://stackoverflow.com/a/55315285
    """
    def __init__(cls, name, bases, attrs):
        errors = []
        for base_cls in bases:
            for meth_name in getattr(base_cls, "__abstractmethods__", ()):
                orig_argspec = inspect.getfullargspec(getattr(base_cls, meth_name))
                target_argspec = inspect.getfullargspec(getattr(cls, meth_name))
                if orig_argspec != target_argspec:
                    errors.append(
                        f"Subclass `{cls.__name__}` of `{base_cls.__name__}` not implemented with correct signature "
                        f"in abstract method {meth_name!r}.\n"
                        f"Expected: {orig_argspec}\n"
                        f"Got: {target_argspec}\n")
        if errors:
            raise TypeError("\n".join(errors))
        super().__init__(name, bases, attrs)


class AbstractGenerator(PrintMixin, metaclass=AbstractSignatureChecker):
    """
    Abstract class for the equation generators.
    """
    def __init__(self, rng=None):
        self.graph = None            # Represents the graph structure
        self.equation = None         # Represents the generated equation
        self.variables = []          # Represents the variables involved in the equation
        self.expression_str = ""     # Represents the equation string
        self.expression_latex = ""   # Represents the latex string

        self.x_data = None           # Input data
        self.y_data = None           # Output data

        self.rng = rng

    @abstractmethod
    def __call__(self, num_variables: int=5, max_terms: int=3,
                 num_realizations: int=10, real_numbers_realizations: bool=True,
                 allowed_operations: list=None, keep_graph: bool=True,
                 keep_data: bool=False, sample_interval: list=[-10, 10],
                 nan_threshold: float=0.1, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls all method that lead to a realization."""
        pass

    @abstractmethod
    def generate_random_graph(self, num_variables: int) -> None:
        """Generates a complex hierarchical random graph."""
        pass

    @staticmethod
    def _make_equation(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            expression = func(*args, **kwargs)
            y = sp.symbols('y')
            equation = sp.Eq(y, expression)
            return equation
        return wrapper

    @abstractmethod
    def generate_equation(self, max_terms: int, allowed_operations: list=None, **kwargs) -> None:
        """Generates an equation that will be applied as a functional mechanism."""
        pass

    @abstractmethod
    def generate_data(self, num_realizations: int, real_numbers_realizations:
                      bool=True, sample_interval: list=[-10, 10]) -> None:
        """Generates a dataset based on the random graph."""
        pass

    @abstractmethod
    def evaluate_equation(self) -> tuple[np.ndarray, np.ndarray]:
        """ This method indexes the currently generated data based on the used
        variables in the sampled equation. This way we can reuse the same data
        for multiple generated equations from the same graph, hence increasing
        efficiency."""
        pass

    @staticmethod
    @abstractmethod
    def get_mantissa_exp(x_data: np.ndarray, y_data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        pass

