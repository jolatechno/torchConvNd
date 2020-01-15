from torchvision import datasets, transforms
import torch

import numpy as np

loss = torch.nn.MSELoss()

def costFunc(out, target):
	result = out.squeeze(0).squeeze(0)
	return loss(target, result).unsqueeze(0).unsqueeze(0).float()


def load_mnist(download=True):
	data = datasets.MNIST('./data', download=download,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		]))
    
	X, Y = [x[0] for x in data], [x[1] for x in data]
    
	X = torch.cat([x for x in X], 0)
	Y = torch.tensor(np.array([np.eye(10)[i] for i in Y]))

	return X, Y

def load_fashion_mnist(download=True):
	data = datasets.FashionMNIST('./data', download=download,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		]))
    
	X, Y = [x[0] for x in data], [x[1] for x in data]
    
	X = torch.cat([x for x in X], 0)
	Y = torch.tensor(np.array([np.eye(10)[i] for i in Y]))

	return X, Y