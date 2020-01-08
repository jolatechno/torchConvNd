from .utils.utils import *
from .convNdFunc import convNdFunc, ConvNdFunc

from torch import nn

"""
n-D convolutional layer with automatic output shape
"""

def convNdAuto(input, output_shape, func, kernel, dilation=1, padding_mode='constant', padding_value=0, *args):
	stride, padding = autoStridePad(input.shape, output_shape, kernel, dilation)
	output_shape = listify(output_shape, input.ndim)

	out = convNdFunc(input, func, kernel, stride, padding, padding_mode, padding_value, *args)

	for i in range(input.ndim):
		out = out.narrow(i, 0, output_shape[i])

	return out

class ConvNdAuto(nn.Module):
	def __init__(self, func, kernel, dilation=1, padding_mode='constant', padding_value=0):
		super(ConvNdAuto, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda input, output_shape, *args: convNdAuto(input, output_shape, func, kernel, dilation, padding_mode, padding_value, *args)

"""
n-D convolution with a recurent function and automatic output shape
"""

def convNdRecAuto(input, mem, output_shape, func, kernel, dilation=1, padding_mode='constant', padding_value=0, *args):
	in_dim = input.ndim
	assert in_dim == mem.ndim, "memory and input dimenssion don't match up"

	padding_mode, padding_value = listify(padding_mode, 2), listify(padding_value, 2)
	output_shape, kernel, dilation = sequencify(output_shape, 2, in_dim), sequencify(kernel, 2, in_dim), sequencify(dilation, 2, in_dim)
	
	stride, padding = autoStridePad(input.shape, output_shape[0], kernel[0], dilation[0])
	mem_stride, mem_padding = autoStridePad(mem.shape, output_shape[1], kernel[1], dilation[1])
	
	pereped, shape = convPrep(input, kernel[0], stride, padding, padding_mode[0], padding_value[0])
	mem_pereped, mem_shape = convPrep(input, kernel[1], mem_stride, mem_padding, padding_mode[1], padding_value[1])
	result, mem_result = func(pereped, mem_pereped, *args)

	if result.ndim == in_dim:
		result = convPost(result, shape)

	if mem_result.ndim == in_dim:
		mem_result = convPost(result_mem, mem_shape)

	for i in range(input.ndim):
		result = result.narrow(i, 0, output_shape[0][i])
		mem_result = mem_result.narrow(i, 0, output_shape[1][i])

	return result, mem_result

class ConvNdRecAuto(nn.Module):
	def __init__(self, func, kernel, dilation=1, padding_mode='constant', padding_value=0):
		super(ConvNdRecAuto, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda input, mem, output_shape, *args: convNdRecAuto(input, mem, output_shape, func, kernel, dilation, padding_mode, padding_value, *args)
