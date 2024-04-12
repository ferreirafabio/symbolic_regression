
from typing import Callable, Union, Optional
import torch


# https://github.com/matkoniecz/math_exam_generator


# TODO:
def sample_random_equation(rng: torch.RandomState) :
    return E

# TODO:
def func_E_latex(E) -> str:
    return latex


def func_E_py(E) -> Callable:
    return python_function

def func_py_tbl(python_function: Callable, seed: int, input_data: Optional[torch.Tensor]) -> torch.Tensor:

    # 1. sample input data if no input_data
    # 2. apply python_function
    sample_matrix = torch.zeros(10, 10)
    return sample_matrix

def func_latex_py(latex: str) -> Callable:
    return python_function


def sanity_check(latex: str, E) -> bool:
    # check if the string is a valid latex equation
    # check if equation matches the number of input variables in E
    # is sanity check is differentiable, it's an auxillary loss
    return True

tokenizer
# https://huggingface.co/witiko/mathberta


model






random_equation(None) -> E
func_e_latex(E) -> latex
func_E_py(E) -> python_function
func_py_tbl(python_function) -> numpy_array, input_data

latex_pred =  model(numpy_array)
loss_train = CE(latex_pred, latex)

func_latex_py(latex_pred) -> python_function_pred
func_py_tbl(python_function_pred, input_data) -> numpy_array_pred
loss_eval = MSE(numpy_array_pred, numpy_array)
