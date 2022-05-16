# Program demonstrating change of variable from a 1D uniform distribution to a 1D gaussian distribution - using composite planar flow

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.optim as optim
from torch.autograd.functional import jacobian

import matplotlib.pyplot as plt
import seaborn as sns
import sys
from time import time


# p(x) - Gaussian
px = tdist.Normal(loc=3., scale=0.5)

# data
num_samples = 10000
x = px.sample((num_samples,))

# q(z) - Uniform
# qz = tdist.Uniform(-10, 10)

# q(z) - standard normal
qz = tdist.Normal(loc=0., scale=1.)

# planar transform
class PlanarTransform(nn.Module):
    def __init__(self, init_sigma=0.01):
        super(PlanarTransform, self).__init__()
        # paramters of the planar transform
        self.w = nn.Parameter(torch.randn(1).normal_(0, init_sigma))
        self.u = nn.Parameter(torch.randn(1).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))
        # non-linearity (only LeakyReLU supported for now)
        self.h = nn.LeakyReLU(negative_slope=0.2)

    # transform function
    def forward(self, z, sum_log_det):
        lin = torch.dot(self.w, z) + self.b
        # formualtion to constrain torch.dot(w,u) > -1 (required for invertibility - refer appendix [A.1] of paper https://arxiv.org/abs/1505.05770)
        wtu = torch.dot(self.w, self.u)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        uhat = self.u + ((m_wtu - wtu) * self.w) / (torch.dot(self.w, self.w))
        # the transform
        x = z + uhat * self.h(lin)
        # log_det_jacobian
        dh = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0
        log_det = torch.log(torch.abs(1 + dh(lin) * torch.dot(self.w, uhat)))
        sum_log_det += log_det
        return x, sum_log_det

    # inverse transform function (inspired from https://github.com/VincentStimper/normalizing-flows/)
    def inverse(self, x):
        lin = torch.dot(self.w, x) + self.b
        wtu = torch.dot(self.w, self.u)
        a = ((lin + self.b)/(1 + wtu) < 0) * (self.h.negative_slope - 1.0) + 1.0 # accounting for slope of LeakyReLU (with the indicator function)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        uhat = self.u + ((m_wtu - wtu) * self.w) / (torch.dot(self.w, self.w))
        uhat = a * uhat
        z = x - (1 / (1 + wtu) * (lin + uhat * self.b))
        return z

# instantiate composite planar flow
flow_length = 3
is_cuda = False
if is_cuda:
    transforms_list = [PlanarTransform().to('cuda:0') for _ in range(flow_length)]
else:
    transforms_list = [PlanarTransform() for _ in range(flow_length)]

# composite flow forward
def composite_flow_forward(z):
    sum_log_det = 0
    for pt in transforms_list:
        z, sum_log_det = pt.forward(z, sum_log_det)
    return z, sum_log_det

# composite flow inverse
def composite_flow_inverse(x):
    for pt in reversed(transforms_list):
        x = pt.inverse(x)
    return x

# loss function
def lossfn(x, qz, is_cuda):
    N = num_samples
    loss = 0
    for i in range(N):
        xi = x[i].unsqueeze(0)
        if is_cuda:
            xi = xi.to('cuda:0')
        z = composite_flow_inverse(xi)
        # jacobian
        # jacob = jacobian(planar_flow_forward, z, create_graph=True)
        # jacob = jacob.squeeze()
        _, sum_log_det = composite_flow_forward(z)
        loss += -qz.log_prob(z.squeeze()) + sum_log_det
    return loss

# optimizer
params = []
for pt in transforms_list:
    params = params + [param for param in pt.parameters()]
optimizer =  optim.Adam(params, lr=1e-2)

# train
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = lossfn(x, qz, is_cuda)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f}')


# plot converged density

z_samples = qz.sample((num_samples,))
xhat_samples = []
for z_sample in z_samples:
    z_sample = z_sample.unsqueeze(0)
    if is_cuda:
        z_sample = z_sample.to('cuda:0')
    xhat_sample, _ = composite_flow_forward(z_sample)
    xhat_samples.append(xhat_sample.squeeze().data.cpu())
xhat_samples = np.array(xhat_samples)
z_samples = z_samples.data.cpu().numpy()

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
sns.distplot(a=z_samples, label='source_distribution')
ax1.legend()
ax2 = fig.add_subplot(2,1,2)
sns.distplot(a=xhat_samples, label='transformed_distribution')
sns.distplot(a=x, label='target_distribution')
ax2.legend()
plt.savefig('composite_planar_flow_gauss2gauss_flowLength='+str(flow_length)+'.png')
