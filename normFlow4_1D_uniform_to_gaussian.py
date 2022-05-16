# Program demonstrating change of variable from a 1D uniform distribution to a 1D gaussian distribution

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.optim as optim
from torch.autograd.functional import jacobian

import matplotlib.pyplot as plt
import seaborn as sns


# p(x) - Gaussian
px = tdist.Normal(loc=3, scale=0.5)

# data
num_samples = 1000
x = px.sample((num_samples,))

# q(z) - Uniform
qz = tdist.Uniform(0, 1)

# parameters of the transform function (to be learnt)
w1 = torch.tensor(1., requires_grad=True)
b1 = torch.tensor(1., requires_grad=True)
w2 = torch.tensor(1., requires_grad=True)
b2 = torch.tensor(1., requires_grad=True)

# utility function - sigmoid
def sigmoid(x):
    return 1/(1+torch.exp(-x))

# utility function - inverse of sigmoid
def inv_sigmoid(x):
    return -torch.log( (1/x)-1 )

# transform function - must approximate the inverse of the cdf of the gaussian (after training is converged)
def transform_fn(z):
    z = inv_sigmoid(z)
    z = w1*z + b1
    z = inv_sigmoid(z)
    x = w2*z + b2
    return x

# inverse of the transform function
def inv_transform_fn(x):
    x = (x-b2)/w2
    x = sigmoid(x)
    x = (x-b1)/w1
    z = sigmoid(x)
    return z

# loss function
def lossfn(x, qz):
    N = num_samples
    loss = 0
    for i in range(N):
        z = inv_transform_fn(x[i])
        # jacobian
        jacob = jacobian(transform_fn, z, create_graph=True)
        loss += -qz.log_prob(z) + torch.log(torch.abs(torch.abs(jacob)))
    return loss

# optimizer
optimizer =  optim.Adam([w1, b1, w2, b2], lr=1e-2)

# train
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = lossfn(x, qz)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch:{epoch} \t loss:{loss:.3f}')

# plot converged density

z_samples = qz.sample((num_samples,))
xhat_samples = transform_fn(z_samples)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
sns.distplot(a=z_samples.data.numpy())
ax2 = fig.add_subplot(2,1,2)
sns.distplot(a=xhat_samples.data.numpy())
plt.show()
