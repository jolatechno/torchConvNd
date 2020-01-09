from .utils.utils import *

from torch import nn
import numpy as np

"""
n-D convolution (or transpose convolution if we use stride_transpose instead of stride) with arbitrary function
"""

def convNdFunc(x, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
	filled = x.clone()
	for dim, s in enumerate(listify(stride_transpose, x.ndim)):
		filled = filled.repeat_interleave(s, dim)

	padded = pad(filled, padding, padding_mode, padding_value)
	strided = view(padded, kernel, stride, dilation)
	inter, batch_shape = strided.flatten(0, x.ndim - 1), strided.shape[:x.ndim]
	result = func(inter, *args)

	return result.reshape(*batch_shape)

class ConvNdFunc(nn.Module):
	def __init__(self, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0):
		super(ConvNdFunc, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, *args: convNdFunc(x, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)