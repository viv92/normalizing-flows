# Program demonstrating change of variable for linear transform on a 1D gaussian - as a first attempt on normalizing flows

import numpy as np
import torch
import torch.distributions as tdist
import torch.optim as optim

# p(x)
px = tdist.Normal(loc=3, scale=5)

# data
num_samples = 1000
x = px.sample((num_samples,))

# q(z)
qz = tdist.Normal(loc=0, scale=1)

# params of the transform function (to be learnt)
params = torch.tensor([1., 1.], requires_grad=True)

# transform function (linear function)
def T(z, params):
    return params[0] + z*params[1]

# loss function --- try matrix implementation of loss fuction in second attempt
def lossfn(x, qz, T, params):
    # z = T_inverse(x) --- try using torch.inverse in second attempt
    z = (x - params[0]) / params[1]
    loss = 0
    N = x.shape[0]
    loss = -torch.sum(qz.log_prob(z)) + N * torch.log(torch.abs(params[1])) # --- try using torch to calcualate jacobian in second attempt
    return loss

# optimizer
optimizer =  optim.Adam([params], lr=0.01)

# train
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = lossfn(x, qz, T, params)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'epoch:{epoch} \t loss:{loss:.3f}')

print(f'converged transform params: {params}')
