# Program implementing Glow for image data
# TODOs:
#1. implement multiscale architecture with squeeze and split layers at each level L (after a flow depth K)
#2. check correcteness of actnorm (normalization across batches or normalization across channels with batch_size=1)

import torch
import numpy as np
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import time


class TinyResnet(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        self.l1 = nn.Linear(z_dim, h_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(h_dim, z_dim) # todo: add a weight normalization layer
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.xavier_normal_(self.l2.weight)

    def forward_prop(self, z):
        a = self.relu(self.l1(z))
        out = self.relu(z + self.l2(a))
        return out


class AffineCoupling(nn.Module):
    def __init__(self, z_dim, bmask):
        super().__init__()
        self.z_dim = z_dim
        self.bmask = bmask
        self.s = TinyResnet(z_dim, z_dim*2)
        self.t = TinyResnet(z_dim, z_dim*2)

    def forward(self, z):
        z1 = self.bmask * z
        z2 = (1 - self.bmask) * z
        x1 = z1
        multiplier = torch.exp(self.s.forward_prop(z1))
        x2 = z2 * multiplier + self.t.forward_prop(z1)
        x = x1 + x2
        log_det_jacobian = torch.sum(torch.abs(torch.sum(torch.log(multiplier), dim=1))) # this sum is over the entire minibatch
        return x, log_det_jacobian

    def inverse(self, x):
        x1 = self.bmask * x
        x2 = (1 - self.bmask) * x
        z1 = x1
        z2 = (x2 - self.t.forward_prop(x1)) * torch.exp(-self.s.forward_prop(x1))
        z = z1 + z2
        return z


class ReverseOrderFlow(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.permute = torch.arange(z_dim-1, -1, -1)
        self.sort = torch.argsort(self.permute)

    def forward(self, z):
        return z[:, self.permute]

    def inverse(self, x):
        return x[:, self.sort]


class BatchNormFlow(nn.Module):
    def __init__(self, z_dim, eps, momentum):
        super().__init__()
        self.batch_mean = torch.zeros(z_dim)
        self.batch_std = torch.ones(z_dim)
        self.running_mean = torch.zeros(z_dim)
        self.running_std = torch.ones(z_dim)
        self.training = True
        self.eps = eps
        self.momentum = momentum

    def calculate_batch_statistics(self, x_batch):
        self.batch_mean = x_batch.mean(0)
        self.batch_std = (x_batch - self.batch_mean).pow(2).mean(0).sqrt() + self.eps
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
        self.running_std = self.momentum * self.running_std + (1 - self.momentum) * self.batch_std

    def forward(self, z):
        if self.training:
            mean = self.batch_mean
            std = self.batch_std
        else:
            mean = self.running_mean
            std = self.running_std
        x = z * std + mean
        batch_size = z.shape[0]
        log_det_jacobian = torch.abs(torch.sum(torch.log(std))) * batch_size
        return x, log_det_jacobian

    def inverse(self, x):
        if self.training:
            self.calculate_batch_statistics(x) # batch statistics make sense to be calculated in the inverse direction / flow
            mean = self.batch_mean
            std = self.batch_std
        else:
            mean = self.running_mean
            std = self.running_std
        z = (x - mean) / std
        return z


class RealNVPFlow(nn.Module):
    def __init__(self, z_dim, flow_length):
        super().__init__()
        self.z_dim = z_dim
        self.flow_length = flow_length
        self.bmask = torch.ones(z_dim)
        self.reverse_block = ReverseOrderFlow(self.z_dim)
        self.batchnorm_blocks = nn.ModuleList()
        self.coupling_blocks = nn.ModuleList()
        self.init_bmask()
        self.init_blocks()

    def init_bmask(self):
        self.bmask = torch.cat((torch.ones(int(self.z_dim/2)), torch.zeros(int(self.z_dim/2))), dim=0)

    def init_blocks(self):
        for i in range(self.flow_length):
            self.batchnorm_blocks.append(BatchNormFlow(self.z_dim, 1e-5, 0.95))
            self.coupling_blocks.append(AffineCoupling(self.z_dim, self.bmask))

    def composite_flow_forward(self, z):
        zt = z
        sum_log_det_jacobian = 0
        for t in np.arange(0, self.flow_length, 1):
            zt = self.reverse_block.forward(zt)
            zt, log_det_jacobian_coupling = self.coupling_blocks[t].forward(zt)
            zt, log_det_jacobian_batchNorm = self.batchnorm_blocks[t].forward(zt)
            sum_log_det_jacobian += log_det_jacobian_batchNorm + log_det_jacobian_coupling
        x = zt
        return x, sum_log_det_jacobian

    def composite_flow_inverse(self, x):
        zt = x
        for t in np.arange(self.flow_length-1, -1, -1):
            zt = self.batchnorm_blocks[t].inverse(zt)
            zt = self.coupling_blocks[t].inverse(zt)
            zt = self.reverse_block.inverse(zt)
        z = zt
        return z



# utility function to plot converged density (contour plots)
def get_plot(epoch):
    # testing - turn off training mode in batchnorm_blocks
    for t in range(flow_real_nvp.flow_length):
        flow_real_nvp.batchnorm_blocks[t].training = False

    z_samples = qz.sample((num_samples,))
    if is_cuda:
        z_samples = z_samples.to('cuda:0')
    xhat_samples, _ = flow_real_nvp.composite_flow_forward(z_samples)
    xhat_samples = xhat_samples.data.cpu().numpy()
    x_samples = X.data.cpu().numpy()
    z_samples = z_samples.data.cpu().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    sns.kdeplot(z_samples[:,0], z_samples[:,1], label='source_distribution')
    ax2 = fig.add_subplot(2,1,2)
    sns.kdeplot(x_samples[:,0], x_samples[:,1], label='target_distribution', color='black')
    sns.kdeplot(xhat_samples[:,0], xhat_samples[:,1], label='transformed_distribution', color='green')
    ax1.legend()
    ax2.legend()
    plt.savefig(foldername + '/' + str(epoch) + '.png')

    # turn back on the training mode in batchnorm_blocks for further training epochs
    for t in range(flow_real_nvp.flow_length):
        flow_real_nvp.batchnorm_blocks[t].training = True


### main

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

is_cuda = False

# target density
px = MultivariateNormal(loc = torch.Tensor([3., 3.]), scale_tril = torch.Tensor([[.5, 0.], [0., .5]]))
# source density
qz = MultivariateNormal(loc = torch.Tensor([0., 0.]), scale_tril = torch.Tensor([[1., 0], [0., 1.]]))

# data
num_samples = 10000
X = px.sample((num_samples,))

num_epochs = 500
batch_size = 1000

# instantiate RealNVPFlow
flow_length = 4
flow_real_nvp = RealNVPFlow(X.shape[1], flow_length)

# loss function
def lossfn(x_batch):
    loss = 0
    z = flow_real_nvp.composite_flow_inverse(x_batch)
    xhat, log_det_jacobian = flow_real_nvp.composite_flow_forward(z)
    a = -torch.sum(qz.log_prob(z))
    b = log_det_jacobian
    loss += a + b
    return loss, a, b

# optimizer
lr=1e-1
optimizer = torch.optim.Adam(params=flow_real_nvp.parameters(), lr=lr)

# folder for saving results
foldername = 'normFlow8_RealNVP_flowLength=' + str(flow_length) + '_lr=' + str(lr) + '_bz=' + str(batch_size)
if not os.path.isdir(foldername):
    os.makedirs(foldername)

# train
loss_qz = []
loss_jacob = []
loss_total = []
num_batches = int(num_samples/batch_size)
for epoch in range(num_epochs):
    batch_index = epoch % num_batches
    # load minibatch
    x_batch = X[batch_index*batch_size : (batch_index+1)*batch_size]
    # loss
    loss, a, b = lossfn(x_batch)
    loss_qz.append(a.data.numpy())
    loss_jacob.append(b.data.numpy())
    loss_total.append(loss.data.numpy())
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch:{} \t loss:{:.3f}\n'.format(epoch, loss.data))
    if epoch % 50 == 9:
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
