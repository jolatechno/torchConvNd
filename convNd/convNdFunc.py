from utils import *

from torch import nn
import numpy as np

"""
n-D convolution with arbitrary function
"""

def generalConvNd(input, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0):
	in_dim = input.ndim

	pereped, shape = convPrep(input, kernel, stride, padding, padding_mode, padding_value)
	result = func(pereped)

	if result.ndim == in_dim:
		return result
	return convPost(result, shape)

class GeneralConvNd(nn.Module):
	def __init__(self, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0):
		super(GeneralConvNd, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda input: generalConvNd(input, func, kernel, stride, padding, padding_mode, padding_value)

"""
n-D convolution with a recursive function
"""

def recConvNd(input, mem, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0):
	in_dim = input.ndim
	assert in_dim == mem.ndim, "memory and input dimenssion don't match up"

	padding_mode, padding_value = listify(padding_mode, 2), listify(padding_value, 2)
	kernel, stride, padding = sequencify(kernel, 2, in_dim), sequencify(stride, 2, in_dim), sequencify(padding, 2, in_dim)

	pereped, shape = convPrep(input, kernel[0], stride[0], padding[0], padding_mode[0], padding_value[0])
	mem_pereped, mem_shape = convPrep(input, kernel[1], stride[1], padding[1], padding_mode[1], padding_value[1])
	result, mem_result = func(pereped, mem_pereped)

	if result.ndim == in_dim:
		result = convPost(result, shape)

	if mem_result.ndim == in_dim:
		mem_result = convPost(result_mem, mem_shape)

	return result, mem_result

class RecConvNd(nn.Module):
	def __init__(self, func, kernel,  stride=1, padding=0, padding_mode='constant', padding_value=0):
		super(RecConvNd, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda input, mem: generalConvNd(input, func, kernel, stride, padding, padding_mode, padding_value)