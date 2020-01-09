from .utils.utils import *
from .convNdFunc import convNdFunc, ConvNdFunc, convTransposeNdFunc, ConvTransposeNdFunc

from torch import nn
import numpy as np

"""
n-D convolutional layer
"""

def convNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0):
	def func(x):
		flat = x.flatten()
		return nn.functional.linear(flat)
		

	return convNdFunc(x, func, kernel, stride, dilation, padding, padding_mode, padding_value)

class ConvNd(nn.Module):
	def __init__(self, kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0):
		super(ConvNd, self).__init__()

		model = nn.Sequential(
			Flatten(),
			nn.Linear(np.prod(kernel), 1, bias))

		conv = ConvNdFunc(model, kernel, stride, dilation, padding, padding_mode, padding_value)
		self.forward = conv.forward
		self.parameters = conv.parameters

"""
n-D transposed convolutional layer
"""

def convTransposeNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0):
	dilation = listify(dilation, x.ndim)

	def func(x):
		result = nn.functional.linear(flat)
		return result.reshape(*kernel)

	return convTransposeNdFunc(x, func, kernel, stride, dilation, padding, padding_mode, padding_value)

class ConvTransposeNd(nn.Module):
	def __init__(self, kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0):
		super(ConvTransposeNd, self).__init__()

		model = nn.Sequential(
			Flatten(),
			nn.Linear(np.prod(kernel), 1, bias))

		conv = ConvTransposeNdFunc(model, kernel, stride, dilation, padding, padding_mode, padding_value)
		self.forward = conv.forward
		self.parameters = conv.parameters