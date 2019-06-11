import sys
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot
import pdb
import time

# timing decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f s' % \
                  (method.__name__, (te - ts)))
        return result
    return timed


transformer = transforms.ToTensor()
MNIST_train = torchvision.datasets.MNIST(root = './Data', train=True, transform=transformer, download=True)
MNIST_test = torchvision.datasets.MNIST(root = './Data', train = False, transform=transformer, download=True)

train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size = 256, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size = 512, shuffle = True, num_workers = 2)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 4)
        self.conv2 = nn.Conv2d(20, 50, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.fc1 = nn.Linear(8 * 8 * 50, 512)
        self.fc2 = nn.Linear(512, 16)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 8 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 16)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class VAE(nn.Module):

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        self.decoder = nn.Sequential(nn.Linear(8, 512), nn.ReLU(),nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 28 * 28), nn.Sigmoid())

    def forward(self, x):

    	#encoding the parameters for latent variables
        z_params = self.encoder(x)
        z_mu = torch.index_select(z_params, 1, torch.arange(0,8).to(device))
        z_logsigma = torch.index_select(z_params, 1, torch.arange(8,16).to(device))

        #random noize
        epsilon = torch.randn((x.shape[0],8))
        epsilon = epsilon.to(device)

        #sample of latent variable
        z = z_mu + torch.exp(z_logsigma) * epsilon

        #probabilities of x according to decoder
        p_x_given_z = self.decoder(z)

        return epsilon, z_logsigma, z, p_x_given_z

    def generate(self):
    	'''
		Samples the random value from latent space and returns
		the corresponding decoded image
    	'''
    	m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(8), torch.eye(8))
    	z = m.sample()
    	with torch.no_grad():
    		generated_image = self.decoder(z)
    	return generated_image





def elbo_loss(xb, epsilon, z_logsigma, z, p_x_z):
	'''
	Returns -elbo assuming the standard normal distribution of latent variables
	'''
	latent_dim = z.shape[1]
	std_norm = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(latent_dim).to(device), torch.eye(latent_dim).to(device))


	x = xb.view(-1, 28*28).round()
	p = p_x_z.view(-1, 28*28)
	LLpxz_matrix = x * torch.log(p)  +  (1 - x) * torch.log(1- p)	
	LLpxz = LLpxz_matrix.mean(1)
	LLq_z = std_norm.log_prob(epsilon) - torch.sum(z_logsigma, 1)
	LLp_z = std_norm.log_prob(z)

	elbo_matrix = LLpxz + LLp_z - LLq_z
	elbo = elbo_matrix.mean()

	return -elbo


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(xb, *model(xb))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


@timeit
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
        	loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

encoder = Encoder()
model = VAE(encoder)


opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# opt = torch.optim.Adagrad(model.parameters())
loss_func = elbo_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)



def preprocess(x, y):
    return x.view(-1, 1, 28, 28).round().to(device), y.to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


train_loader = WrappedDataLoader(train_loader, preprocess)
test_loader = WrappedDataLoader(test_loader, preprocess)

num_epochs = int(sys.argv[1])

fit(num_epochs, model, loss_func, opt, train_loader, test_loader)

torch.save(model, 'VAEmodel.pt')
