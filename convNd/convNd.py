from utils import *
from convNdFunc import *

from torch import nn
import numpy as np

"""
n-D convolutional layer
"""

def convNd(input, weight, kernel, stride=1, dilation=1, padding=0, bias=None):
	dilation = listify(dilation, input.ndim)

	def func(x):
		flat = x.flatten()
		result = nn.functional.linear(flat)
		if dilation == 1:
			return result
		return result.reshape(*dilation)

	return generalConvNd(input, func, kernel, stride, dilation, padding)

class ConvNd(nn.Module):
	def __init__(self, kernel, stride=1, dilation=1, padding=0, bias=False):
		super(ConvNd, self).__init__()
		
		model = nn.Linear(np.prod(kernel), np.prod(dilation), bias)
		self.parameters = model.parameters

		dilation = listify(dilation, input.ndim)

		def func(x):
			flat = x.flatten()
			result = model(flat)
			if dilation == 1:
				return result
			return result.reshape(*dilation)

		conv = GeneralConvNd(func, kernel, stride, dilation, padding)
		self.forward = conv.forward