# Program implementing Neural Autoregressive Flow (NAF)
# note that NAF is used only for density esimation and not as a generative model.
# so we only model the x -> z direction
# not sure if the z -> x direction is mathematically tractable in NAF since the neural net (DDSF) increases the dimensionality of input (from scalar input to vector hidden layer)

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os

#torch.autograd.set_detect_anomaly(True)



# MADE network - outputs the weights for DDSF (in one shot for all x_dims)
class MADE(nn.Module):
    def __init__(self, x_dim, ddsf_num_layers):
        super().__init__()
        self.x_dim = x_dim
        self.weights_eta_w = nn.Parameter(torch.Tensor(x_dim, ddsf_num_layers, x_dim, x_dim))
        self.weights_eta_u = nn.Parameter(torch.Tensor(x_dim, ddsf_num_layers, x_dim, x_dim))
        self.weights_a = nn.Parameter(torch.Tensor(x_dim, ddsf_num_layers, x_dim, x_dim))
        self.weights_b = nn.Parameter(torch.Tensor(x_dim, ddsf_num_layers, x_dim, x_dim))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.weights_eta_w.data)
        nn.init.xavier_normal_(self.weights_eta_u.data)
        nn.init.xavier_normal_(self.weights_a.data)
        nn.init.xavier_normal_(self.weights_b.data)

    def calculate_ddsf_weights(self, x):
        eta_w = (self.weights_eta_w.tril(-1) @ x).T # eta_w.shape = [x_dim, ddsf_num_layers, x_dim] since eta_w[i] is a [ddsf_num_layers, x_dim] dimensional matrix constituting weights of a DDSF_Net dependent autoregressively on x[1:i-1]
        eta_u = (self.weights_eta_u.tril(-1) @ x).T
        a = (self.weights_a.tril(-1) @ x).T
        b = (self.weights_b.tril(-1) @ x).T
        return eta_w, eta_u, a, b



# Deep Dense Sigmoidal Flow - a single block / net of the NAF - takes input a single dimension of x at a time : DDSF(x_d)
class DDSF(nn.Module):
    def __init__(self, x_dim, num_layers):
        super().__init__()
        self.x_dim = x_dim
        self.num_layers = num_layers

        self.softmax = nn.Softmax(dim=0)
        self.softplus = nn.Softplus()
        self.u0 = self.softmax(torch.ones(x_dim))
        self.wz = self.softmax(torch.ones(x_dim))
        for l in range(self.num_layers):
            self.w = self.softmax(torch.ones((self.num_layers, self.x_dim, self.x_dim)))
            self.u = self.softmax(torch.ones((self.num_layers, self.x_dim, self.x_dim)))
            self.a = self.softplus(torch.ones(self.num_layers, self.x_dim))
            self.b = torch.ones(self.num_layers, self.x_dim)

        self.z_d = torch.Tensor(x_dim)
        self.x_d = torch.Tensor(x_dim)

    def update_params(self, eta_w, eta_u, a, b):
        self.u0 = self.softmax(eta_u[0])
        self.wz = self.softmax(eta_w[self.num_layers-1])
        for l in range(self.num_layers):
            self.w[l] = self.softmax(self.w[l] + eta_w[l])
            self.u[l] = self.softmax(self.u[l] + eta_u[l])
            self.a[l] = self.softplus(a[l])
            self.b[l] = b[l]

    def forward(self, z_d):
        raise NotImplementedError('Transform in the z -> x direction is not mathematically tractable for DDSF.')

    def inverse(self, x_d): # according to appendix C.1 in paper
        x_d.requires_grad_()
        self.x_d = x_d
        h = self.u0 * x_d
        for l in range(self.num_layers):
            C = (self.a[l] * (self.u[l] @ h)) + self.b[l]
            D = self.w[l] @ torch.sigmoid(C)
            h = torch.logit(D)
        z = torch.dot(self.wz, h)
        self.z_d = z
        return z

    def calculate_gradient_dzdx(self):
        # returns the gradient d(z_d)/d(x_d) - used to calculate log_det_jacobian later in NAF

        # implemented using torch.autograd.functional.jacobian
        # log_det_jacobian = torch.log(torch.abs(torch.det(jacobian(self.inverse, x_d, retain_graph=False))))

        # implemented using torch.autograd.grad
        grad_dzdx = torch.autograd.grad(self.z_d, self.x_d, retain_graph=False)
        return grad_dzdx



# permute flow
class PermuteFlow(nn.Module):
    def __init__(self, x_dim):
        super(PermuteFlow, self).__init__()
        self.permute = torch.randperm(x_dim)
        self.sort = torch.argsort(self.permute)

    def forward(self, z):
        # return z[:self.permute]
        return z[self.permute]

    def inverse(self, z):
        # return z[:, self.sort]
        return z[self.sort]



# Neural Autoregressive Flow - multiple instances of DDSF operating autoregressively on all dimensions of input x
class NAFlow(nn.Module):
    def __init__(self, x_dim, num_layers):
        super().__init__()
        self.x_dim = x_dim
        self.num_layers = num_layers
        self.CNet = MADE(self.x_dim, self.num_layers)
        self.DDSNet_blocks = nn.ModuleList()
        self.init_DDSF_blocks()
        # self.permute_block = PermuteFlow(self.x_dim)

    def init_DDSF_blocks(self):
        for d in range(self.x_dim):
            self.DDSNet_blocks.append(DDSF(self.x_dim, self.num_layers))

    def flow_forward(self, z0):
        raise NotImplementedError('Transform in the z -> x direction is not mathematically tractable for DDSF.')

    def flow_inverse(self, x):
        log_det_jacob = 0
        eta_w, eta_u, a, b = self.CNet.calculate_ddsf_weights(x)
        for d in range(self.x_dim):
            self.DDSNet_blocks[d].update_params(eta_w[d], eta_u[d], a[d], b[d])
            z_d = self.DDSNet_blocks[d].inverse(x[d])
            z_d = z_d.unsqueeze(0)
            if d == 0:
                z = z_d
            else:
                z = torch.cat((z, z_d))
            grad_dzdx = self.DDSNet_blocks[d].calculate_gradient_dzdx()[0]
            log_det_jacob += torch.log(torch.abs(grad_dzdx))
        return z, log_det_jacob




# utility function to plot converged density (contour plots)
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

batch_size = 1000
num_epochs = 500
lr = 1e-2

# instantiate NAFlow
ddsf_num_layers = 2
flow = NAFlow(X.shape[1], ddsf_num_layers)


# loss function
def lossfn(x_batch):
    loss = 0
    for x in x_batch:  ### TODO: Vectorized implementation instead of looping over datapoints
        z, log_det_jacob = flow.flow_inverse(x)
        a = -qz.log_prob(z)
        b = -log_det_jacob # note the negative sign - since we use grad_dzdx and not grad_dxdz here
        loss += a + b
        #print(f'a:{a.data} \t b:{b.data} \t loss:{loss.data}')
    return loss, a, b

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

foldername = 'normFlow9_NAF'
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
    loss_total.append(loss.data.numpy())
    loss_qz.append(a.data.numpy())
    loss_jacob.append(b.data.numpy())
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()
    if epoch % 1 == 0:
        #print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f} \t time_taken:{et-st:.3f}')
        print(f'epoch:{epoch} \t loss:{loss.data:.3f}')
        # get_plot(epoch)

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
