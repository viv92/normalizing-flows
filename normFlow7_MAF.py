# Program implementing Masked Autoregressive Flow (MAF)


import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os

#torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

is_cuda = False

# data

px = MultivariateNormal(loc=torch.Tensor([3., 3.]), scale_tril=torch.Tensor([[.5, 0],[0, .5]])) # target distribution

qz = MultivariateNormal(loc=torch.Tensor([0., 0.]), scale_tril=torch.Tensor([[1., 0],[0, 1.]])) # source distribution

#qz = tdist.Uniform(torch.Tensor([0, 0]), torch.Tensor([1, 1]))

# px = tdist.Normal(loc=3., scale=0.5)
# qz = tdist.Normal(loc=0., scale=1.)

num_samples = 10000
X = px.sample((num_samples,))
if is_cuda:
    X = X.to('cuda:0')



# Masked Autoregressive Network - constitutes a single autoregressive transform block
class MaskedLinearAR(nn.Module):
    def __init__(self, z_dim):
        super(MaskedLinearAR, self).__init__()
        self.z_dim = z_dim
        self.weights_mu = nn.Parameter(torch.Tensor(z_dim, z_dim))
        self.bias_mu = nn.Parameter(torch.Tensor(z_dim))
        self.weights_sigma = nn.Parameter(torch.Tensor(z_dim, z_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(z_dim))
        self.mu = torch.zeros(z_dim)
        self.sigma = torch.ones(z_dim) # assuming diagonal covariance matrix for now
        if is_cuda:
            self.mu = self.mu.to('cuda:0')
            self.sigma = self.sigma.to('cuda:0')
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.weights_mu.data)
        self.bias_mu.data.uniform_(0, 1)
        nn.init.xavier_normal_(self.weights_sigma.data)
        self.bias_sigma.data.uniform_(0, 1)

    def calculate_mu(self, z):
        m = (z @ self.weights_mu.tril(-1)) + self.bias_mu # tril(-1) yeilds the lower triangular matrix with zero diagonal - inducing the autoregressive operation 
        return m

    def calculate_sigma(self, z):
        s = (z @ self.weights_sigma.tril(-1)) + self.bias_sigma
        #sigma = self.sigmoid(s)
        sigma = torch.exp(s)
        return sigma

    def forward(self, z):
        # mu = self.calculate_mu(z)
        # sigma = self.calculate_sigma(z)
        z_next = (self.sigma * z) + self.mu
        log_det_jacobian = torch.abs(torch.sum(torch.log(self.sigma)))
        return z_next, log_det_jacobian

    def inverse(self, z):
        self.mu = self.calculate_mu(z)
        self.sigma = self.calculate_sigma(z)
        z_prev = (z - self.mu) / self.sigma
        return z_prev


# permute flow
class PermuteFlow(nn.Module):
    def __init__(self, z_dim):
        super(PermuteFlow, self).__init__()
        self.permute = torch.randperm(z_dim)
        self.sort = torch.argsort(self.permute)

    def forward(self, z):
        # return z[:self.permute]
        return z[self.permute]

    def inverse(self, z):
        # return z[:, self.sort]
        return z[self.sort]



# Inverse Autoregressive Flow - compositions of autoregressive transforms
class MAFlow(nn.Module):
    def __init__(self, flow_length, x_dim):
        super(MAFlow, self).__init__()
        self.z_dim = x_dim
        self.flow_length = flow_length
        self.MA_blocks = nn.ModuleList()
        self.init_MA_blocks()
        self.permute_block = PermuteFlow(self.z_dim)

    def init_MA_blocks(self):
        for i in range(self.flow_length):
            self.MA_blocks.append(MaskedLinearAR(self.z_dim))

    def composite_flow_forward(self, z0):
        zt = z0
        sum_log_det_jacobian = 0
        for ar in self.MA_blocks:
            zt = self.permute_block.forward(zt)
            zt, log_det_jacobian = ar.forward(zt)
            sum_log_det_jacobian += log_det_jacobian
            # zt = self.permute_block.forward(zt)
        x = zt
        return x, sum_log_det_jacobian

    def composite_flow_inverse(self, x):
        zt = x
        for ar in reversed(self.MA_blocks):
            zt = ar.inverse(zt)
            zt = self.permute_block.inverse(zt)
        z0 = zt
        return z0


def get_plot(epoch):
    # plot converged density (contour plots)
    z_samples = qz.sample((num_samples,))
    if is_cuda:
        z_samples = z_samples.to('cuda:0')
    xhat_samples = []
    for z in z_samples:
        xhat, _ = flow.composite_flow_forward(z)
        xhat_samples.append(xhat.data.cpu().numpy())
    x_samples = X.data.cpu().numpy()
    z_samples = z_samples.data.cpu().numpy()
    xhat_samples = np.array(xhat_samples)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    sns.kdeplot(z_samples[:,0], z_samples[:,1], label='source_distribution')
    ax2 = fig.add_subplot(2,1,2)
    sns.kdeplot(x_samples[:,0], x_samples[:,1], label='target_distribution', color='black')
    sns.kdeplot(xhat_samples[:,0], xhat_samples[:,1], label='transformed_distribution', color='green')
    ax1.legend()
    ax2.legend()
    plt.savefig(foldername + '/' + str(epoch) + '.png')


# instantiate MAFlow
flow_length = 3
flow = MAFlow(flow_length=flow_length, x_dim=X.shape[1])
if is_cuda:
    flow = flow.to('cuda:0')

# loss function
def lossfn(X, batch_size):
    # indices = torch.randperm(X.shape[0])
    # X = X[indices]
    loss = 0
    for i in range(batch_size):  ### TODO: Vectorized implementation instead of looping over datapoints
        x = X[i]
        z = flow.composite_flow_inverse(x)
        _, sum_log_det_jacobian = flow.composite_flow_forward(z)
        #loss += -qz.log_prob(z) + sum_log_det_jacobian

        a = -qz.log_prob(z)
        b = sum_log_det_jacobian
        loss += a + b
        #print(f'a:{a.data} \t b:{b.data} \t loss:{loss.data}')
    return loss, a, b

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)

foldername = 'normFlow7_MAF_flowLength=' + str(flow_length)
if not os.path.isdir(foldername):
    os.makedirs(foldername)

# train
loss_qz = []
loss_jacob = []
loss_total = []

num_epochs = 500
batch_size = int(num_samples/10)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss, a, b = lossfn(X, batch_size)
    loss_total.append(loss.data.numpy())
    loss_qz.append(a.data.numpy())
    loss_jacob.append(b.data.numpy())
    loss.backward(retain_graph=False)
    optimizer.step()
    if epoch % 50 == 0:
        #print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f} \t time_taken:{et-st:.3f}')
        print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f}')
        get_plot(epoch)

# loss curve
fig = plt.figure()
plt.plot(loss_total, label='loss_total')
plt.plot(loss_qz, label='loss_qz')
plt.plot(loss_jacob, label='loss_jacob')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.title('loss curve')
plt.savefig(foldername + '/loss_curve.png')
