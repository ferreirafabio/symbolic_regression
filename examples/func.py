import numpy as np
import sympy as sp
from sympy import latex
sp.init_printing()

from gpr.synthetic.noise_scale import init_noise_dist
from gpr.synthetic.utils import sample_recursive_scm
from gpr.synthetic import GraphModel, MechanismModel


class RandomTree(GraphModel):

    def __call__(self, rng, n_vars):
        # generate a random spanning tree
        mat = np.zeros((n_vars, n_vars))
        vertices = {rng.integers(n_vars)}
        for _ in range(n_vars - 1):
            u = rng.choice(list(vertices))
            for v in rng.permutation(n_vars):
                if v not in vertices:
                    mat[u, v] = 1
                    vertices.add(v)
                    break
        return mat


class SinusoidalAdditive(MechanismModel):
    """
    Sinusoidal mechanism with additive noise
        f_j = c sin(w.x) + b
    """

    def __init__(self,
                 scale,
                 weight,
                 bias,
                 noise,
                 noise_scale=None,
                 noise_scale_heteroscedastic=None,
                 n_interv_vars=0,
                 interv_dist=None):

        assert interv_dist is not None or n_interv_vars == 0

        self.scale = scale
        self.weight = weight
        self.bias = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    def __call__(self, rng, g, n_observations_obs, n_observations_int):

        # construct mechanism for each node
        n_vars = g.shape[-1]
        f = []
        for j in range(n_vars):
            # sample parameters
            # each call may use random hyperparameters, so call once per node
            w = self.weight(rng, shape=(n_vars,))
            c = self.scale(rng, shape=(1,))
            b = self.bias(rng, shape=(1,))
            print(w, c, b)

            # bind parameters to mechanism function of node j
            f.append(lambda x, is_parent, z, param=w, sc=c, bias=b: sc * np.sin(x @ (param * is_parent)) + bias + z)

        # construct noise distribution for each node
        nse = []
        for j in range(n_vars):
            # sample parameters and bind to sampling function
            # each call may sample a random noise scale, so call once per node
            nse.append(init_noise_dist(rng=rng,
                                       dim=g[:, j].sum(),
                                       dist=self.noise,
                                       noise_scale=self.noise_scale,
                                       noise_scale_heteroscedastic=self.noise_scale_heteroscedastic))

        # sample recursively over g given functionals and noise distributions for all requested intervention settings
        data = sample_recursive_scm(
            rng=rng,
            n_observations_obs=n_observations_obs,
            n_observations_int=n_observations_int,
            g=g,
            f=f,
            nse=nse,
            interv_dist=self.interv_dist,
            n_interv_vars=self.n_interv_vars,
        )
        return data


class SinusoidalAdditiveFixed(MechanismModel):
    """
    Sinusoidal mechanism with additive noise
        f_j = c sin(w.x) + b
    """

    def __init__(self,
                 scale,
                 weight,
                 bias,
                 noise,
                 noise_scale=None,
                 noise_scale_heteroscedastic=None,
                 n_interv_vars=0,
                 interv_dist=None):

        assert interv_dist is not None or n_interv_vars == 0

        self.scale = scale
        self.weight = weight
        self.bias = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    @staticmethod
    def get_mechanism_sympy():
        var = sp.symbols('sin(2x)')
        noise_var = sp.symbols('epsilon')
        return latex(2 * var + 1 + noise_var)

    def __call__(self, rng, g, n_observations_obs, n_observations_int):

        # construct mechanism for each node
        n_vars = g.shape[-1]
        f = []
        w = np.asarray([2] * n_vars)
        c = np.asarray([2])
        b = np.asarray([1])
        for j in range(n_vars):
            # bind parameters to mechanism function of node j
            f.append(lambda x, is_parent, z, param=w, sc=c, bias=b: sc * np.sin(x @ (param * is_parent)) + bias + z)

        # construct noise distribution for each node
        nse = []
        for j in range(n_vars):
            # sample parameters and bind to sampling function
            # each call may sample a random noise scale, so call once per node
            nse.append(init_noise_dist(rng=rng,
                                       dim=g[:, j].sum(),
                                       dist=self.noise,
                                       noise_scale=self.noise_scale,
                                       noise_scale_heteroscedastic=self.noise_scale_heteroscedastic))

        # sample recursively over g given functionals and noise distributions for all requested intervention settings
        data = sample_recursive_scm(
            rng=rng,
            n_observations_obs=n_observations_obs,
            n_observations_int=n_observations_int,
            g=g,
            f=f,
            nse=nse,
            interv_dist=self.interv_dist,
            n_interv_vars=self.n_interv_vars,
        )
        return data
