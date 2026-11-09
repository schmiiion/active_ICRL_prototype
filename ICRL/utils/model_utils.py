import torch
from torch import nn


def create_constraint_mlp(input_dim, output_dim, hidden_sizes, activation='relu'):
    """Build Liu's Linear/ReLU MLP with PyTorch's original initialization.

    This matches the benchmark's SB3 ``create_mlp``. OmniSafe's factory
    reinitializes each Linear weight after construction, which changes both
    seeded parameters and the RNG state despite using the same distribution.
    The caller appends Sigmoid (MECL) or Softplus (VICRL).
    """
    layers = []
    activation_class = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'elu': nn.ELU}[activation]
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(last_dim, hidden_dim), activation_class()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return layers


def dirichlet_kl_divergence_loss(alpha, prior):
    """
    KL divergence between two dirichlet distribution
    The mean is alpha/(alpha+beta) and variance is alpha*beta/(alpha+beta)^2*(alpha+beta+1)
    There are multiple ways of modelling a dirichlet:
    1) by Laplace approximation with logistic normal: https://arxiv.org/pdf/1703.01488.pdf
    2) by directly modelling dirichlet parameters: https://arxiv.org/pdf/1901.02739.pdf
    code reference：
    1） https://github.com/sophieburkhardt/dirichlet-vae-topic-models
    2） https://github.com/is0383kk/Dirichlet-VAE
    """
    analytical_kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
    analytical_kld += torch.sum(torch.lgamma(prior), dim=1)
    analytical_kld -= torch.sum(torch.lgamma(alpha), dim=1)
    minus_term = alpha - prior
    # tmp = torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    digamma_term = torch.digamma(alpha) - \
                   torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    test = torch.sum(torch.mul(minus_term, digamma_term), dim=1)
    analytical_kld += test
    # self.analytical_kld = self.mask * self.analytical_kld  # mask paddings
    return analytical_kld
