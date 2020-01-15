from torchvision import datasets
import torch

def load_mnist():
	return datasets.MNIST('./data', train=True, transform=None, target_transform=None, download=True)

mnist = load_mnist()