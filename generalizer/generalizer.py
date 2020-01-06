from utils import *
from torch import nn

import numpy as np

"""
n-D convolutional layer
"""

def convNd(input, weight, kernel, dilation=1, padding=0, bias=None):
	preped = convPrep(input, kernel, dilation, padding)
	shape = preped.shape[:len(preped.shape)]
	size = np.prod(preped.shape[len(preped.shape):])
	result = nn.functional.linear(preped.reshape(-1, size), weight, bias)
	return result.reshape(shape)

class ConvNd(nn.Modules):
	def __init__(self, kernel, dilation=1, padding=0, bias=False):
		self.Fprep = ConvPrep(kernel, dilation, padding)
		self.linear = nn.Linear(np.prod(kernel), np.prod(dilation), bias)

	def forward(self, input):
		preped = self.Fprep(input)
		shape = preped.shape[:len(preped.shape)]
		size = np.prod(preped.shape[len(preped.shape):])
		result = self.linear(preped.reshape(-1, size), weight, bias)
		return result.reshape(shape)

"""
n-D convolution with arbitrary function
"""

class Generalizer(nn.Modules):
	def __init__(self, func, kernel, dilation=1, padding=0, initializer=False, args={,}):
		self.net = net
		self.parameters = net.parameters

		pass
		
	def forward(self, input, cost=0):
		pass