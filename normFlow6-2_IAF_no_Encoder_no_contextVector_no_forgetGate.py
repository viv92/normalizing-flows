# Program implementing Variational Inference using Inverse Autoregressive Flow (IAF) (https://arxiv.org/abs/1606.04934)


import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

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
        m = (z @ self.weights_mu.tril(-1)) + self.bias_mu
        return m

    def calculate_sigma(self, z):
        s = (z @ self.weights_sigma.tril(-1)) + self.bias_sigma
        sigma = self.sigmoid(s)
        return sigma

    def forward(self, z):
        # mu = self.calculate_mu(z)
        # sigma = self.calculate_sigma(z)
        z_next = (self.sigma * z) + self.mu
        log_det_jacobian = torch.sum(torch.log(self.sigma))
        return z_next, log_det_jacobian

    def inverse(self, z):
        self.mu = self.calculate_mu(z)
        self.sigma = self.calculate_sigma(z)
        z_prev = (z - self.mu) / self.sigma
        return z_prev



# Inverse Autoregressive Flow - compositions of autoregressive transforms
class IAFlow(nn.Module):
    def __init__(self, flow_length, x_dim):
        super(IAFlow, self).__init__()
        self.z_dim = x_dim
        self.flow_length = flow_length
        self.IAF_blocks = nn.ModuleList()
        self.init_IAF_blocks()

    def init_IAF_blocks(self):
        for i in range(self.flow_length):
            self.IAF_blocks.append(MaskedLinearAR(self.z_dim))

    def composite_flow_forward(self, z0):
        zt = z0
        sum_log_det_jacobian = 0
        for ar in self.IAF_blocks:
            zt, log_det_jacobian = ar.forward(zt)
            sum_log_det_jacobian += log_det_jacobian
        x = zt
        return x, sum_log_det_jacobian

    def composite_flow_inverse(self, x):
        zt = x
        for ar in reversed(self.IAF_blocks):
            zt = ar.inverse(zt)
        z0 = zt
        return z0


# instantiate IAFlow
flow_length = 3
flow = IAFlow(flow_length=flow_length, x_dim=X.shape[1])
if is_cuda:
    flow = flow.to('cuda:0')

# loss function
def lossfn(X, batch_size):
    # indices = torch.randperm(X.shape[0])
    # X = X[indices]
    loss = 0
    for i in range(batch_size): # this is wrong implementation of using batching - we are only using the first batch for all training epochs  
        x = X[i]
        z = flow.composite_flow_inverse(x)
        _, sum_log_det_jacobian = flow.composite_flow_forward(z)
        #loss += -qz.log_prob(z) + sum_log_det_jacobian

        a = -qz.log_prob(z)
        b = sum_log_det_jacobian
        loss += a + b
        #print(f'a:{a.data} \t b:{b.data} \t loss:{loss.data}')
    return loss

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)

# train
num_epochs = 500
batch_size = int(num_samples/10)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = lossfn(X, batch_size)
    # if loss < 0:
    #     break
    loss.backward(retain_graph=False)
    optimizer.step()
    if epoch % 50 == 0:
        #print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f} \t time_taken:{et-st:.3f}')
        print(f'epoch:{epoch} \t loss:{loss.squeeze().data.cpu():.3f}')

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

#print('z_samples.shape:{} x_samples.shape:{} xhat_samples.shape:{}'.format(z_samples.shape, x_samples.shape, xhat_samples.shape))

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
sns.kdeplot(z_samples[:,0], z_samples[:,1], label='source_distribution')

ax2 = fig.add_subplot(2,1,2)
sns.kdeplot(x_samples[:,0], x_samples[:,1], label='target_distribution', color='black')
sns.kdeplot(xhat_samples[:,0], xhat_samples[:,1], label='transformed_distribution', color='green')

ax1.legend()
ax2.legend()
plt.savefig('normFlow6-2_flowLength='+str(flow_length)+'.png')
