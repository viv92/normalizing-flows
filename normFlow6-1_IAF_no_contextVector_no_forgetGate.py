# Program implementing Inverse Autoregressive Flow (IAF) (https://arxiv.org/abs/1606.04934)
# Implementation inspired from https://github.com/acids-ircam/pytorch_flows/flows_04.ipynb

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


# Masked Autoregressive Network - constitutes a single autoregressive transform block
class MaskedLinearAR(nn.Module):
    def __init__(self, z_dim):
        super(MaskedLinearAR, self).__init__()
        self.z_dim = z_dim
        self.weights_mu = nn.Parameter(torch.Tensor(z_dim*2, z_dim)) # input dim = z_dim*2 since input = [z,h]
        self.bias_mu = nn.Parameter(torch.Tensor(z_dim))
        self.weights_sigma = nn.Parameter(torch.Tensor(z_dim*2, z_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(z_dim))
        self.mu = torch.Tensor(z_dim)
        self.sigma = torch.Tensor(z_dim) # assuming diagonal covariance matrix for now
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal(self.weights_mu.data)
        self.bias_mu.data.uniform_(-1, 1)
        nn.init.xavier_normal(self.weights_sigma.data)
        self.bias_sigma.data.uniform_(-1, 1)

    def forward(self, z):
        m = (z @ self.weights_mu.tril(-1)) + self.bias_mu
        s = (z @ self.weights_sigma.tril(-1)) + self.bias_sigma
        sigma = nn.Sigmoid(s)
        self.mu = m
        self.sigma = sigma
        z_next = (sigma * z) + ((1-sigma) * m)
        return z_next

    def inverse(self, z):
        z_prev = (z - ((1-self.sigma) * self.mu)) / self.sigma
        return z_prev

    def log_det_jacobain(self):
        return torch.sum(torch.log(self.sigma))


# Encoder network - for the first transform (parameterizes the source distribution z0 ~ q(z0|x) = Gaussian(z0 | mu(theta(x)), sigma(theta(x))) )
class Encoder(nn.Module):
    def __init__(self, x_dim):
        super(Encoder, self).__init__()
        self.z_dim = x_dim
        self.fc_mu = nn.Linear(x_dim, z_dim)
        self.fc_sigma = nn.Linear(x_dim, z_dim)
        self.fc_context_h = nn.Linear(x_dim, z_dim)
        self.activation = nn.ReLU()
        self.mu = torch.Tensor(z_dim)
        self.sigma = torch.Tensor(z_dim) # assuming diagonal covariance matrix for now
        self.eps = torch.Tensor(z_dim)

    def forward(self, x):
        mu = self.activation(self.fc_mu(x))
        sigma = self.activation(self.fc_sigma(x))
        context_h = self.activation(self.fc_context_h(x))
        eps = torch.FloatTensor(z_dim).normal_() # sample epsilon from standard gaussian
        z0 = sigma * eps + mu
        return z0, context_h

    def logqz0(self):
        return ((-self.z_dim/2.)*(torch.log(2*torch.pi))) - (torch.dot(self.eps, self.eps)/2.) - torch.sum(torch.log(self.sigma))



# Inverse Autoregressive Flow - compositions of autoregressive transforms
class IAFlow(nn.Module):
    def __init__(self, flow_length):
        super(IAFlow, self).__init__()
