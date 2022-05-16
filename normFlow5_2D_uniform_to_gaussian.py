# Program demonstrating change of variable from a 2D uniform distribution to a 2D gaussian distribution

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.optim as optim
from torch.autograd.functional import jacobian
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt
import seaborn as sns


# p(x) - Gaussian
px = MultivariateNormal(loc=torch.Tensor([3., 3.]), scale_tril=torch.Tensor([[2., 0],[0, 2.]]))

# data
num_samples = 10000
x = px.sample((num_samples,))

# q(z) - Uniform
qz = tdist.Uniform(torch.Tensor([0, 0]), torch.Tensor([1, 1]))

# parameters of the transform function (to be learnt)
w1 = torch.tensor([1., 1.], requires_grad=True)
b1 = torch.tensor([1., 1.], requires_grad=True)
w2 = torch.tensor([1., 1.], requires_grad=True)
b2 = torch.tensor([1., 1.], requires_grad=True)

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
        jacob = jacob.squeeze()
        loss += torch.sum(-qz.log_prob(z)) + torch.log(torch.abs(torch.det(jacob)))
    return loss

# optimizer
optimizer =  optim.Adam([w1, b1, w2, b2], lr=1e-2)

# train
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = lossfn(x, qz)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch:{epoch} \t loss:{loss:.3f}')

# plot converged density (contour plots)

z_samples = qz.sample((num_samples,))
xhat_samples = transform_fn(z_samples)
z_samples = z_samples.data.numpy()
xhat_samples = xhat_samples.data.numpy()

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
sns.kdeplot(z_samples[:,0], z_samples[:,1])
ax2 = fig.add_subplot(2,1,2)
sns.kdeplot(xhat_samples[:,0], xhat_samples[:,1])
plt.show()
