from .utils.utils import *
from .convNdFunc import convNdFunc, ConvNdFunc

from torch import nn
import numpy as np

"""
n-D convolutional layer
"""

def convNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0):
	def func(x):
		flattend = x.flatten(1, -1)
		if bias is not None:
			return flattend @ weight + bias
		return flattend @ weight

	return convNdFunc(x, func, kernel, stride, dilation, padding, 1, padding_mode, padding_value)

class ConvNd(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0):
		super(ConvNd, self).__init__()

		model = nn.Sequential(
			Flatten(),
			nn.Linear(np.prod(kernel)*in_channels, out_channels, bias))

		conv = ConvNdFunc(model, kernel, stride, dilation, padding, 1, padding_mode, padding_value)
		self.forward = conv.forward
		self.parameters = conv.parameters

"""
n-D transposed convolutional layer
"""

def convTransposeNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0):
	def func(x):
		flattend = x.flatten(1, -1)
		if bias is not None:
			return flattend @ weight + bias
		return flattend @ weight

	return convNdFunc(x, func, kernel, 1, dilation, padding, stride, padding_mode, padding_value)

class ConvTransposeNd(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0):
		super(ConvTransposeNd, self).__init__()

		model = nn.Sequential(
			Flatten(),
			nn.Linear(np.prod(kernel)*in_channels, out_channels, bias))

		conv = ConvNdFunc(model, kernel, 1, dilation, padding, stride, padding_mode, padding_value)
		self.forward = conv.forward
		self.parameters = conv.parameters