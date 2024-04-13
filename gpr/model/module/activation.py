import torch
from torch import nn


@torch.jit.script
def golu_fwd(x):
    z = x * torch.exp(-1 * torch.exp(-1 * x))
    return z.to(dtype=x.dtype)


@torch.jit.script
def golu_bwd(g, x):
    z = g * (torch.exp(-torch.exp(-x)) + x * torch.exp(-x) * torch.exp(-torch.exp(-x)))
    return z.to(dtype=x.dtype)


class FastGoLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return golu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = golu_bwd(grad_output, input)
        return tmp


class GoLU(nn.Module):
    """
    GoLU Activation Function
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Calls the forward pass of the GoLU Activation Function

        Returns:
            torch.Tensor: The output of the forward method
        """
        return self.forward(*args)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GoLU Activation Function

        Args:
            z (torch.Tensor): The input tensor on which the activation is computed

        Returns:
            torch.Tensor: The activated tensor
        """
        return FastGoLUFunction.apply(z)


class Activation(nn.Module):
    def __init__(self, act_fn_name: str, is_glu: bool):
        super().__init__()

        self.is_glu = is_glu

        if "silu" in act_fn_name:
            self.act = nn.SiLU()
        elif "relu" in act_fn_name:
            self.act = nn.ReLU()
        elif "gelu" in act_fn_name:
            self.act = nn.GELU(approximate='tanh')
        elif "golu" in act_fn_name:
            self.act = GoLU()

    def forward(self, x: torch.Tensor):

        if self.is_glu:
            gate_states, up_states = x.chunk(2, dim=-1)
            return self.act(gate_states) * up_states
        else:
            return self.act(x)
