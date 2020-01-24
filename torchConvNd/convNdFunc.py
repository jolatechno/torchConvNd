from .utils.utils import *

from torch import nn
import numpy as np

"""
n-D convolution (or transpose convolution if we use stride_transpose instead of stride) with arbitrary function
"""

def convNdFunc(x, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
	filled = x.clone()
	
	kernel = [-1, -1] + listify(kernel, x.ndim - 2)
	stride = [1, 1] + listify(stride, x.ndim - 2)
	dilation = [1, 1] + listify(dilation, x.ndim - 2)
	padding = [0, 0] + listify(padding, x.ndim - 2)
	stride_transpose = listify(stride_transpose, x.ndim - 2)

	for dim, s in enumerate(stride_transpose):
		if s != 1:
			filled = filled.repeat_interleave(s, dim + 2)

	padded = pad(filled, padding, padding_mode, padding_value)
	strided = view(padded, kernel, stride, dilation)

	batch_length, batch_shape = strided.shape[0], strided.shape[2:x.ndim]

	flattend = strided.flatten(2, x.ndim - 1)
	permuted = flattend.permute(0, 2, 1, *range(3, x.ndim + 1))
	flattend = permuted.flatten(0, 1)
	result = func(flattend, *args)

	if isinstance(result, tuple):
		result, additional = result[0], result[1:]
		reshaped = result.reshape(batch_length, *batch_shape, -1)
		permuted = reshaped.permute(0, -1, *range(1, x.ndim - 1))
		return (permuted,) + additional

	reshaped = result.reshape(batch_length, *batch_shape, -1)
	permuted = reshaped.permute(0, -1, *range(1, x.ndim - 1))
	return permuted

class ConvNdFunc(nn.Module):
	def __init__(self, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0):
		super(ConvNdFunc, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, *args: convNdFunc(x, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)