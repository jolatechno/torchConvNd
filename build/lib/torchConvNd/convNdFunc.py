from .utils.utils import *

from torch import nn
import numpy as np

"""
n-D convolution with arbitrary function
"""

def convNdFunc(input, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0, *args):
	in_dim = input.ndim

	pereped, shape = convPrep(input, kernel, stride, padding, padding_mode, padding_value)
	result = func(pereped, *args)

	if result.ndim == in_dim:
		return result
	return convPost(result, shape)

class ConvNdFunc(nn.Module):
	def __init__(self, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0):
		super(ConvNdFunc, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda input, *args: convNdFunc(input, func, kernel, stride, padding, padding_mode, padding_value, *args)