# Program implementing Glow for toy 2-d data
# the actnorm in this implementation is essentially batchnorm - since we are dealing with 2-d data with no channels, in contrast to the case of images with rgb channels
# for the same reason, we do not have the squeeze and split levels

# check glow_images for the version with multi_scale architecture including the squeeze and split layers at each level L, after a flow depth of K

# So this implementation of Glow = RealNVP_2d + 1x1 Conv (learnable permutation transform)
# todo: check if the assumption in this implementation actnorm = batchnorm is correct or instead, actnorm = normalize across z_dim with batch_size = 1

## some possible mistakes in real_nvp implementation realized from this implementation of glow:
#1. log_det_jacobian = torch.sum(torch.log(torch.abs())) and not torch.abs(torch.sum(torch.log()))
#2. mutli-scale architecture - the implementation of realNVP_2d does not match with figure 4.b in the realNVP paper - code up realNVP_images which will include this feature

import torch
import numpy as np
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import time


class DNN(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.l1 = nn.Linear(z_dim, h_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(h_dim, z_dim) # todo: add a weight normalization layer
        self.init_weights()

    def init_weights(self):
        # # zero initialization (section 3.3 of glow paper)
        # self.l1.weight.data.fill_(0.0)
        # self.l2.weight.data.fill_(0.0)

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
        self.s = DNN(z_dim, z_dim*2)
        self.t = DNN(z_dim, z_dim*2)

    def forward(self, z):
        z1 = self.bmask * z
        z2 = (1 - self.bmask) * z
        x1 = z1
        multiplier = torch.exp(self.s.forward_prop(z1))
        x2 = z2 * multiplier + self.t.forward_prop(z1)
        x = x1 + x2
        # log_det_jacobian = torch.sum(torch.sum(torch.abs(torch.log(multiplier)), dim=1))
        log_det_jacobian = torch.sum(torch.sum(torch.log(torch.abs(multiplier)), dim=1)) # this sum is over the entire minibatch
        return x, log_det_jacobian

    def inverse(self, x):
        x1 = self.bmask * x
        x2 = (1 - self.bmask) * x
        z1 = x1
        z2 = (x2 - self.t.forward_prop(x1)) * torch.exp(-self.s.forward_prop(x1))
        z = z1 + z2
        return z


class ActNormFlow(nn.Module): # todo : check if actnorm = batchnorm is correct or instead, actnorm = normalize across z_dim with batch_size = 1
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
        # log_det_jacobian = torch.sum(torch.abs(torch.log(std))) * batch_size
        log_det_jacobian = torch.sum(torch.log(torch.abs(std))) * batch_size
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



class Conv1x1_Flow(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # initial weight / permutation matrix
        W_init = torch.linalg.qr(torch.randn(z_dim,z_dim))[0]
        self.W = W_init
        # LU decomposition of W
        P, L, U = torch.lu_unpack(*torch.lu(W_init))
        self.P = torch.Tensor(P) # P matrix is constant throughout training
        # declare L, U and s as learnable parameters
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.s = nn.Parameter(torch.Tensor(z_dim))

    # converts L and U matrices into L', U' and s, where
    # L' is lower triangular with ones on diagonal
    # U' is upper triangular with zeros on diagonal
    # s is vector whose entries = diagonal elements of U
    def reformulate_lu(self):
        L, U = self.L.data, self.U.data
        # extract diagonal of L
        diag_L_vector = torch.diag(L)
        # divide columns of L to decompose L into L'(with ones on diagonal) and diag(L)
        for d in range(self.z_dim):
            L[:,d] /= diag_L_vector[d]
        # diag(L)
        diag_L_matrix = torch.diag(diag_L_vector)
        # subsume diag(L) into U
        U = diag_L_matrix @ U
        # decompose U to U' and s
        s = torch.diag(U)
        U -= torch.diag(s)
        self.L.data, self.U.data, self.s.data = L, U, s
        self.W = self.P @ self.L @ (self.U + torch.diag(self.s))

    def forward(self, z):
        batch_size = z.shape[0]
        x = z @ self.W
        # log_det_jacobian = torch.sum(torch.abs(torch.log(self.s))) * batch_size
        log_det_jacobian = torch.sum(torch.log(torch.abs(self.s))) * batch_size
        return x, log_det_jacobian

    def inverse(self, x):
        # reformulate L, U and s in the desired form - required at the begining of each new epoch (after every gradient step)
        self.reformulate_lu()
        z = x @ torch.inverse(self.W)
        return z



# one step of Glow
class Glow_onestep(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.bmask = torch.cat((torch.ones(int(self.z_dim/2)), torch.zeros(int(self.z_dim/2))), dim=0)
        self.conv1x1 = Conv1x1_Flow(self.z_dim)
        self.actnorm = ActNormFlow(self.z_dim, 1e-5, 0.95)
        self.affine_coupling = AffineCoupling(self.z_dim, self.bmask)

    def forward(self, z):
        log_det_jacobian = 0
        z, log_det_jacobian_coupling = self.affine_coupling.forward(z)
        z, log_det_jacobian_conv1x1 = self.conv1x1.forward(z)
        x, log_det_jacobian_actnorm = self.actnorm.forward(z)
        log_det_jacobian += log_det_jacobian_actnorm + log_det_jacobian_conv1x1 + log_det_jacobian_coupling
        return x, log_det_jacobian

    def inverse(self, x):
        x = self.actnorm.inverse(x)
        x = self.conv1x1.inverse(x)
        z = self.affine_coupling.inverse(x)
        return z



# multi-scale Glow architecture (actually just mutliple steps of glow_onestep for the toy 2d data, without any squeeze and split layers at multiple levels)
class Glow(nn.Module):
    def __init__(self, z_dim, flow_depth_K, num_levels_L='deprecated for glow_2d'):
        super().__init__()
        self.z_dim = z_dim
        self.flow_depth_K = flow_depth_K
        self.glow_blocks = nn.ModuleList()
        self.init_glow_blocks()

    def init_glow_blocks(self):
        for k in range(self.flow_depth_K):
            self.glow_blocks.append(Glow_onestep(self.z_dim))

    def composite_flow_forward(self, z):
        sum_log_det_jacobian = 0
        zt = z
        for k in np.arange(0, self.flow_depth_K, 1):
            zt, log_det_jacobian = self.glow_blocks[k].forward(z)
            sum_log_det_jacobian += log_det_jacobian
        x = zt
        return x, sum_log_det_jacobian

    def composite_flow_inverse(self, x):
        zt = x
        for k in np.arange(self.flow_depth_K-1, -1, -1):
            zt = self.glow_blocks[k].inverse(zt)
        z = zt
        return z



# utility function to plot converged density (contour plots)
def get_plot(epoch):
    # testing - turn off training mode in glow_blocks.actnorm
    for k in range(glow.flow_depth_K):
        glow.glow_blocks[k].actnorm.training = False

    z_samples = qz.sample((num_samples,))
    if is_cuda:
        z_samples = z_samples.to('cuda:0')
    xhat_samples, _ = glow.composite_flow_forward(z_samples)
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

    # turn back on the training mode in glow_blocks.actnorm  for further training epochs
    for k in range(glow.flow_depth_K):
        glow.glow_blocks[k].actnorm.training = True


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

num_epochs = 5000
batch_size = 100

# instantiate glow
flow_depth = 2
glow = Glow(X.shape[1], flow_depth)

# loss function
def lossfn(x_batch):
    loss = 0
    z = glow.composite_flow_inverse(x_batch)
    xhat, log_det_jacobian = glow.composite_flow_forward(z)
    a = -torch.sum(qz.log_prob(z))
    b = log_det_jacobian
    loss += a + b
    return loss, a, b

# optimizer
lr=1e-3
optimizer = torch.optim.Adam(params=glow.parameters(), lr=lr)

# folder for saving results
foldername = 'normFlow10_glow2d_flow_depth=' + str(flow_depth) + '_lr=' + str(lr) + '_bz=' + str(batch_size)
if not os.path.isdir(foldername):
    os.makedirs(foldername)

# train
loss_qz = []
loss_jacob = []
loss_total = []
for epoch in range(num_epochs):
    # load minibatch
    indices = torch.randperm(num_samples)
    X = X[indices]
    x_batch = X[:batch_size]
    # loss
    loss, a, b = lossfn(x_batch)
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

    loss_qz.append(a.data.numpy())
    loss_jacob.append(b.data.numpy())
    loss_total.append(loss.data.numpy())
    if epoch % 10 == 0:
        print('epoch:{} \t loss:{:.3f} \t a:{:.3f} \t b:{:.3f}\n'.format(epoch, loss.data, a.data, b.data))
    if epoch % 500 == 9:
        get_plot(epoch)

# loss curve
fig = plt.figure()
plt.plot(loss_total, label='loss_total')
plt.plot(loss_qz, label='loss_qz')
plt.plot(loss_jacob, label='loss_jacob')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.ylim(-1000, 1000)
plt.title('loss curve')
plt.savefig(foldername + '/loss_curve.png')
