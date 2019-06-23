import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim
import numpy as np
from matplotlib import pyplot
import pdb
import time



class LinEncoder(nn.Module):
    '''
    Convolutional Encoder NN for VAE on MNIST.
    '''
    def __init__(self, dimension):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc21 = nn.Linear(512, dimension)
        self.fc22 = nn.Linear(512, dimension)


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc21(x)
        log_sigma = self.fc22(x)
        return mu, log_sigma



class ConvEncoder(nn.Module):
    '''
    Convolutional Encoder NN for VAE on MNIST.
    '''
    def __init__(self, dimension):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 4)
        self.conv2 = nn.Conv2d(20, 50, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.fc1 = nn.Linear(8 * 8 * 50, 512)
        self.fc21 = nn.Linear(512, dimension)
        self.fc22 = nn.Linear(512, dimension)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x.view(-1, 1, 28, 28))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 8 * 50)
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        log_sigma = self.fc22(x)
        return mu, log_sigma

class Decoder(nn.Module):
    '''
    Simple 2-layer Decoder NN
    '''

    def __init__(self, dimension):
        super().__init__()
        self.fc1 = nn.Linear(dimension, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))



class VAE(nn.Module):
    '''
    '''
    def __init__(self, latent_dim, enc_type = 'conv'):
        super().__init__()
        self.latent_dim = latent_dim

        if enc_type == 'conv':
            self.encoder = ConvEncoder(latent_dim)
        if enc_type == 'lin':
            self.encoder = LinEncoder(latent_dim)

        self.decoder = Decoder(latent_dim)



    def encode(self, x):
        mu, log_sigma = self.encoder(x)
        return mu, log_sigma

    def decode(self, z):
        return self.decoder(z)

    def sample_e_z(self, mu, log_sigma):
        '''
        Samples random normal epsilon and produces Z, given it's mean and log variance
        Returns both epsilon and Z
        '''

        epsilon = torch.randn_like(log_sigma)

        z = mu + torch.exp(0.5*log_sigma) * epsilon

        return epsilon, z


    def forward(self, x):
    	#encoding the parameters for latent variables
        z_mu, z_logsigma = self.encode(x)

        #sampling noize and Z
        epsilon, z = self.sample_e_z(z_mu, z_logsigma)

        #p(x|z) according to decoder
        p_x_given_z = self.decode(z)

        return epsilon, z_logsigma, z, p_x_given_z

    def generate(self):
    	'''
		Samples the random value from latent space and returns
		the corresponding decoded image
    	'''
    	z = torch.randn(64,self.latent_dim).to(device)
    	with torch.no_grad():
    		generated_image = self.decode(z)
    	return generated_image


def elbo_loss(x, epsilon, z_logsigma, z, p_x_z):
    '''
    Returns -elbo assuming the standard normal distribution of latent variables
    '''

    x = x.view(-1, 28*28)
    p = p_x_z.view(-1, 28*28)
    #point-wise bernoulli cross entropy
    LLpxz_matrix = x * torch.log(p)  +  (1 - x) * torch.log(1- p) 

    LLpxz = LLpxz_matrix.sum(1)
    LLq_z = std_norm.log_prob(epsilon) - torch.sum(z_logsigma, 1)
    LLp_z = std_norm.log_prob(z)

    elbo_matrix = LLpxz + LLp_z - LLq_z
    sample_elbo = torch.mean(elbo_matrix)

    return -sample_elbo



def loss_batch(model, loss_func, xb, opt=None):
    loss = loss_func(xb, *model(xb))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, save_path):
    print(f'Epochs to train  :                  {epochs}\n')
    print(f'Model parameters will be saved to:  {save_path}\n{"."*30}')
    
    for epoch in range(epochs):
        model.train()
        ts = time.perf_counter()
        for batch_idx, (xb, _) in enumerate(train_dl):
            loss_batch(model, loss_func, xb, opt)
            
            
        model.eval()
        with torch.no_grad():
            losses = [loss_batch(model, loss_func, xb) for xb, _ in valid_dl]
        val_loss = torch.mean(torch.tensor(losses))

        tf = time.perf_counter()
        print(f'Epoch : {epoch+1}, batches: {batch_idx + 1}, Validation Loss: {val_loss:.2f}, took : {tf - ts:.2f} sec')

        torch.save(model.state_dict(), save_path)
       



def preprocess(x, y):
    return x.to(device), y.to(device)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training  the VAE model on MNIST dataset')
    parser.add_argument('-b', '--batch-size',dest = 'b_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-z','--z-dim',  type=int, default=16, metavar='N',
                        help='Latent space dimensionality (delault: 16)')
    parser.add_argument('--save-model',  type=str, default='VAE_dict.pt',
                        help='Where to store the model (VAE_dict.pt by default)')
    parser.add_argument('--load-model',  type=str, default=None,
                        help='Path to the stored model (a new model is trained by default)')
    parser.add_argument('--encoder',  type=str, default='conv', choices = ['lin', 'conv'],
                        help='Type of Encoder (delault: convolutional)')
    args = parser.parse_args()

    transformer = transforms.ToTensor()
    MNIST_train = torchvision.datasets.MNIST(root = './Data', train=True, transform=transformer, download=True)
    MNIST_test = torchvision.datasets.MNIST(root = './Data', train = False, transform=transformer, download=True)

    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size = args.b_size, shuffle = True, drop_last = True)
    test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size = args.b_size, shuffle = True, drop_last = True)


    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f'Working on :{device}\n{"."*30}')
    model = VAE(args.z_dim, args.encoder)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    std_norm = MultivariateNormal(torch.zeros(args.z_dim), torch.eye(args.z_dim))
    loss_func = elbo_loss

    train_loader = WrappedDataLoader(train_loader, preprocess)
    test_loader = WrappedDataLoader(test_loader, preprocess)

    fit(args.epochs, model, loss_func, opt, train_loader, test_loader, args.save_model)


    model.eval()
    number = model.generate()
    save_image(number.view(64, 1, 28, 28), 'recent_samples.png')





