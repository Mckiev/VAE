import sys
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from VAE import VAE, ConvEncoder, LinEncoder, Decoder
import time


parser = argparse.ArgumentParser(description='Training MNIST classifier on the latent representation learned by VAE')

parser.add_argument('-b', '--batch-size',dest = 'b_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')

parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('-z','--z-dim',  type=int, default=32, metavar='N',
                    help='Latent space dimensionality (delault: 32)')

parser.add_argument('--save-classifier',  type=str, default=None,
                    help='Path to the stored classifier (model is not saved by default)')

parser.add_argument('--load-classifier',  type=str, default=None,
                    help='path to pre-trained classifier (None by default)')

parser.add_argument('--VAE-model',  type=str, default='VAE_dict.pt',
                    help='Path to VAE model. 32 latent dim model stored in \'VAE_dict.pt\' is used by default')

parser.add_argument('--encoder',  type=str, default='conv', choices = ['lin', 'conv'],
                        help='Type of Encoder (delault: convolutional)')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f'Working on :{device}')  

vae_model = VAE(args.z_dim, args.encoder)
vae_model.load_state_dict(torch.load(args.VAE_model))
vae_model.to(device)


transformer = transforms.ToTensor()
MNIST_train = torchvision.datasets.MNIST(root = './Data', train=True, transform=transformer, download=True)
MNIST_test = torchvision.datasets.MNIST(root = './Data', train = False, transform=transformer, download=True)


train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size = args.b_size, shuffle = True, drop_last = True)
test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size = 64, shuffle = True, drop_last = True)


def Encode(x):
    with torch.no_grad():
        mu, _ = vae_model.encode(x.to(device))
    return mu.cpu() 




class Classifier(nn.Module):

    def __init__(self,z_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 64)
        self.fc3 = nn.Linear(64, 10)
      
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return x



def loss_batch(model, xb, yb, opt=None):
    l_func = nn.CrossEntropyLoss()
    loss = l_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


def train(epochs, model, opt, train_dl, valid_dl):
    ts = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_dl):
            loss_batch(model, Encode(xb), yb, opt)
            
            
        model.eval()
        with torch.no_grad():
            losses = [loss_batch(model, Encode(xb), yb) for xb, yb in valid_dl]
        val_loss = torch.mean(torch.tensor(losses))
        print(f'Epoch : {epoch+1} .......  Validation Loss: {val_loss:.4f} ...... time elapsed: {time.time()-ts:.2f} sec')
        model.train()
        if args.save_classifier:
            torch.save(model.state_dict(), args.save_classifier)
    if args.save_classifier:
        print(f'model saved to {args.save_classifier}')

model = Classifier(args.z_dim)  
if args.load_classifier:
    model.load_state_dict(torch.load(args.load_classifier))    

opt = torch.optim.Adagrad(model.parameters())




train(args.epochs, model, opt, train_loader, test_loader)


wrong = 0
total = 0
for x, y in test_loader:
	with torch.no_grad():
		predictions = np.argmax(model(Encode(x)), axis= 1)
		wrong += len(predictions[predictions!=y])
		total += len(y)

print(f'tested : {total}, errors : {wrong}, error rate = {wrong/total}')

