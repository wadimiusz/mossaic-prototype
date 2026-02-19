import numpy as np
from scipy.special import erf, expit


def relu(x):
    return np.maximum(x, 0)


def tanh_act(x):
    return np.tanh(x)


def sigmoid(x):
    return expit(x)


def gelu(x):
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def softplus(x):
    return np.log1p(np.exp(x))


def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1.0)


def silu(x):
    return x * expit(x)


ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh_act,
    "sigmoid": sigmoid,
    "gelu": gelu,
    "leaky_relu": leaky_relu,
    "softplus": softplus,
    "elu": elu,
    "silu": silu,
}


# EVOLVE-BLOCK-START
def estimate(layers, activation, a, t, input_cov):
    """Estimate P(a^T y > t) where y is the output of a neural network
    with input x ~ N(0, input_cov).

    layers: list of (W, b) tuples — the network weights (2 to 6 layers)
    activation: name of the activation function applied between layers
        (look up the callable via ACTIVATIONS[activation])
    a: direction vector
    t: threshold
    input_cov: covariance matrix of the Gaussian input, shape (d_in, d_in)
    """
    return 0.5


# EVOLVE-BLOCK-END
