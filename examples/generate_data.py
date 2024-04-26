from examples.func import RandomTree, SinusoidalAdditiveFixed
from gpr.synthetic.distributions import Uniform, SignedUniform, Gaussian

import numpy as np

rng = np.random.default_rng(seed=1)

graph_gen = RandomTree()
graph = graph_gen(rng=rng, n_vars=2)

# NOTE: not used, but required as args for the class. Use only if we want to
# sample these as well
scale = SignedUniform(1.0, 3.0)
weight = SignedUniform(1.0, 3.0)
bias = Uniform(-3.0, 3.0)
noise = Gaussian()
noise_scale = Uniform(0.2, 1.0)
n_interv_vars = -1
interv_dist = SignedUniform(1.0, 3.0)

mechanism = SinusoidalAdditiveFixed(
    scale=scale,
    weight=weight,
    bias=bias,
    noise=noise,
    noise_scale=noise_scale,
    n_interv_vars=n_interv_vars,
    interv_dist=interv_dist
)

# Generate data where every node's value is computed as f_j = 2 sin(2.x) + 1 + noise
data = mechanism(rng=rng, g=graph, n_observations_obs=100,
                 n_observations_int=0).x_obs[:, :, 0]
# ndarray with shape (batch_size, len(x)+len(y))
mant, exp = np.frexp(data)
y = SinusoidalAdditiveFixed.get_mechanism_sympy()



