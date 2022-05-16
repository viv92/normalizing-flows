# Program implementing Variational Inference using Inverse Autoregressive Flow (IAF) (https://arxiv.org/abs/1606.04934)

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

#torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)

# Masked Autoregressive Network - constitutes a single autoregressive transform block
class MaskedLinearAR(nn.Module):
    def __init__(self, z_dim):
        super(MaskedLinearAR, self).__init__()
        self.z_dim = z_dim
        self.weights_context_h = nn.Parameter(torch.Tensor(z_dim, z_dim))
        self.weights_mu = nn.Parameter(torch.Tensor(z_dim, z_dim))
        self.bias_mu = nn.Parameter(torch.Tensor(z_dim))
        self.weights_sigma = nn.Parameter(torch.Tensor(z_dim, z_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(z_dim))
        self.mu = torch.zeros(z_dim)
        self.sigma = torch.ones(z_dim) # assuming diagonal covariance matrix for now
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.weights_context_h.data)
        nn.init.xavier_normal_(self.weights_mu.data)
        self.bias_mu.data.uniform_(0, 1)
        nn.init.xavier_normal_(self.weights_sigma.data)
        self.bias_sigma.data.uniform_(0, 1)

    def calculate_mu_sigma(self, z, context_h):
        # context_multiplier = torch.exp(context_h @ self.weights_context_h.tril(-1))
        context_multiplier = 1
        m = (z @ self.weights_mu.tril(-1)) * context_multiplier + self.bias_mu
        s = (z @ self.weights_sigma.tril(-1)) * context_multiplier + self.bias_sigma
        sigma = torch.sigmoid(s)
        #sigma = torch.exp(s)
        return m, sigma

    def forward(self, z, context_h):
        m, sigma = self.calculate_mu_sigma(z, context_h)
        self.mu = m
        self.sigma = sigma
        # z_next = (self.sigma * z) + ((1-self.sigma) * self.mu)
        z_next = self.sigma * z + self.mu
        log_det_jacobian = torch.sum(torch.log(self.sigma))
        return z_next, log_det_jacobian

    def inverse(self, z, context_h):
        # m, sigma = self.calculate_mu_sigma(z, context_h)
        # self.mu = m
        # self.sigma = sigma
        # z_prev = (z - ((1-self.sigma) * self.mu)) / self.sigma
        z_prev = (z - self.mu) / self.sigma
        return z_prev


# Encoder network - for the first transform (parameterizes the source distribution z0 ~ q(z0|x) = Gaussian(z0 | mu(theta(x)), sigma(theta(x))) )
class Encoder(nn.Module):
    def __init__(self, x_dim):
        super(Encoder, self).__init__()
        self.z_dim = x_dim
        self.fc_mu = nn.Linear(x_dim, self.z_dim)
        self.fc_sigma = nn.Linear(x_dim, self.z_dim)
        self.fc_context_h = nn.Linear(x_dim, self.z_dim)
        self.mu = torch.zeros(self.z_dim)
        self.sigma = torch.ones(self.z_dim) # assuming diagonal covariance matrix for now
        self.context_h = torch.zeros(self.z_dim)
        self.eps = torch.Tensor(self.z_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.xavier_normal_(self.fc_sigma.weight)
        nn.init.xavier_normal_(self.fc_context_h.weight)

    def forward(self, x):
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))
        context_h = self.fc_context_h(x)
        eps = torch.FloatTensor(self.z_dim).normal_() # sample epsilon from standard gaussian
        z0 = sigma * eps + mu
        self.sigma = sigma
        self.mu = mu
        self.eps = eps
        self.context_h = context_h
        return z0, context_h

    def logqz0(self):
        # a = ((-self.z_dim/2.)*(torch.log(torch.Tensor([2*torch.pi]))))
        # b = (torch.dot(self.eps, self.eps)/2.)
        # print('sigma:{} mu:{} eps:{}'.format(self.sigma.data, self.mu.data, self.eps.data))
        # c = torch.sum(torch.log(self.sigma))
        # print('a:{} b:{} c:{}'.format(a.data, b.data, c.data))
        # return a - b - c
        return ((-self.z_dim/2.)*(torch.log(torch.Tensor([2*torch.pi])))) - (torch.dot(self.eps, self.eps)/2.) - torch.sum(torch.log(self.sigma))



# Inverse Autoregressive Flow - compositions of autoregressive transforms
class IAFlow(nn.Module):
    def __init__(self, flow_length, x_dim):
        super(IAFlow, self).__init__()
        self.z_dim = x_dim
        self.flow_length = flow_length
        self.encoder = Encoder(x_dim)
        self.IAF_blocks = nn.ModuleList()
        self.init_IAF_blocks()

    def init_IAF_blocks(self):
        for i in range(self.flow_length):
            self.IAF_blocks.append(MaskedLinearAR(self.z_dim))

    def composite_flow_forward(self, z0):
        zt = z0
        sum_log_det_jacobian = 0
        for ar in self.IAF_blocks:
            zt, log_det_jacobian = ar.forward(zt, self.encoder.context_h)
            sum_log_det_jacobian += log_det_jacobian
        x = zt
        return x, sum_log_det_jacobian

    def composite_flow_inverse(self, x):
        zt = x
        for ar in reversed(self.IAF_blocks):
            zt = ar.inverse(zt, self.encoder.context_h)
        z0 = zt
        return z0


def get_plot(epoch):
    # plot converged density (contour plots)
    z_samples = []
    for x in X:
        z_sample, context_h = flow.encoder.forward(x)
        z_samples.append(z_sample)
    xhat_samples = []
    for i in range(len(z_samples)):
        z = z_samples[i]
        xhat, _ = flow.composite_flow_forward(z)
        xhat_samples.append(xhat.data.numpy())
        z_samples[i] = z.data.numpy()
    x_samples = X.data.numpy()
    xhat_samples = np.array(xhat_samples)
    z_samples = np.array(z_samples)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    sns.kdeplot(z_samples[:,0], z_samples[:,1], label='source_distribution')
    # sns.kdeplot(data=z_samples, label='source_distribution')
    ax2 = fig.add_subplot(2,1,2)
    sns.kdeplot(x_samples[:,0], x_samples[:,1], label='target_distribution', color='black')
    # sns.kdeplot(data=x_samples, label='target_distribution', color='black')
    sns.kdeplot(xhat_samples[:,0], xhat_samples[:,1], label='transformed_distribution', color='green')
    # sns.kdeplot(data=xhat_samples, label='transformed_distribution', color='green')
    ax1.legend()
    ax2.legend()
    plt.title('Epoch.%i'%(epoch), fontsize=15)
    plt.savefig(foldername + '/' + str(epoch) + '.png')



# data
num_samples = 10000
px = MultivariateNormal(loc=torch.Tensor([3., 3.]), scale_tril=torch.Tensor([[.5, 0],[0, .5]])) # target distribution
X = px.sample((num_samples,))

# instantiate IAFlow
flow_length = 1
flow = IAFlow(flow_length=flow_length, x_dim=X.shape[1])

# loss function
def lossfn(X):
    N = num_samples
    loss = 0
    # calculate loss over the dataset
    for i in range(N):
        x = X[i]

        # call encoder.forward to set the source/reference distribution q(z0)
        z0, context_h = flow.encoder.forward(x)
        logqz0 = flow.encoder.logqz0()

        # now do the flow stuff for the source distribution q(z0) set by the encoder
        #z = flow.composite_flow_inverse(x)
        _, sum_log_det_jacobian = flow.composite_flow_forward(z0)
        a = -logqz0
        b = sum_log_det_jacobian
        loss += a + b
        #print(f'a:{a.data} \t b:{b.data} \t loss:{loss.data}')
    return loss

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

foldername = 'normFlow6_sigmoid_noContext_lowLR_flowLength=' + str(flow_length)
if not os.path.isdir(foldername):
    os.makedirs(foldername)

# train
num_epochs = 200
for epoch in range(num_epochs):
    st = time.time()
    optimizer.zero_grad()
    loss = lossfn(X)
    loss.backward(retain_graph=False)
    optimizer.step()
    et = time.time()
    if epoch % 5 == 0:
        #print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f}')
        print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f} \t time_taken:{et-st:.3f}')
        get_plot(epoch)
