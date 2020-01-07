from .utils.utils import *
from .convNdFunc import generalConvNd, GeneralConvNd

from torch import nn
import numpy as np

"""
n-D convolutional layer
"""

def convNd(input, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0):
	dilation = listify(dilation, input.ndim)

	def func(x):
		flat = x.flatten()
		result = nn.functional.linear(flat)
		return result.reshape(*dilation)

	return generalConvNd(input, func, kernel, stride, dilation, padding, padding_mode, padding_value)

class ConvNd(nn.Module):
	def __init__(self, kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0):
		super(ConvNd, self).__init__()
		
		model = nn.Sequential([Flatten(),
			nn.Linear(np.prod(kernel), np.prod(dilation), bias),
			Reshape(kernel)])

		conv = GeneralConvNd(model, kernel, stride, dilation, padding, padding_mode, padding_value)
		self.forward = conv.forward
		self.parameters = conv.parameters

