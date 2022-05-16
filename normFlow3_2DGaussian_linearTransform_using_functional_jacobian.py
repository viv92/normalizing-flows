# Program demonstrating change of variable for linear transform on a 2D gaussian

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import time

# p(x)
px = MultivariateNormal(loc=torch.Tensor([3., 3.]), scale_tril=torch.Tensor([[4., 0.],[0., 4.]]))

# data
num_samples = 100
x = px.sample((num_samples,))

# q(z)
qz = MultivariateNormal(loc=torch.Tensor([0., 0.]), scale_tril=torch.Tensor([[1., 0.],[0., 1.]]))

# params of the transform function (to be learnt)
params_mu = torch.tensor([1., 1.], requires_grad=True)
# params_mu = params_mu.unsqueeze(0)
params_scale_tril = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)

# transform function (linear function)
def T(z):
    params_cov = (params_scale_tril * torch.transpose(params_scale_tril, 0, 1))
    return params_mu + torch.mm( z, params_cov )

# loss function --- try matrix implementation of loss fuction in second attempt
def lossfn(x, qz, T, params_mu, params_scale_tril):
    N = num_samples
    loss = 0
    for i in range(N):
        # z = T_inverse(x)
        params_cov = (params_scale_tril * torch.transpose(params_scale_tril, 0, 1))
        z = torch.mm( (x[i] - params_mu).unsqueeze(0), torch.inverse(params_cov) )
        # loss += -qz.log_prob(z) + torch.log(torch.abs(torch.det(params_cov))) # --- try using torch to calcualate jacobian in second attempt
        jacob = jacobian(T, z, create_graph=True)
        jacob = jacob.squeeze()
        z = z.squeeze(0)
        loss += -qz.log_prob(z) + torch.log(torch.abs(torch.det(jacob))) # --- using torch.autograd.functional.jacobian to get the jacobian
    return loss

# optimizer
optimizer =  optim.Adam([params_mu, params_scale_tril], lr=0.01)

# train
num_epochs = 1000
st = time.time()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = lossfn(x, qz, T, params_mu, params_scale_tril)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'epoch:{epoch} \t loss:{loss:.3f}')
et = time.time()
print(f'converged transform params_mu: {params_mu}')
print(f'converged transform params_scale_tril: {params_scale_tril}')
print('time taken (functional jacobian): ', et-st)
