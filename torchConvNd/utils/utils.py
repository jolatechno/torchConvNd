import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from collections.abc import Iterable

"""
helper functions
"""

def listify(x, dims=1):
	if dims < 0:
		return x

	if isinstance(x, Iterable) and not isinstance(x, str):
		if len(x) != dims:
			ValueError("Shape don't match up")

		return x

	return [x for i in range(dims)]

"""
functions for convNdAuto
"""

def convShape(input_shape, kernel, stride=1, dilation=1, padding=0, stride_transpose=1):
	if any([isinstance(p, Iterable) for p in [input_shape, kernel, stride, dilation, padding, stride_transpose]]):
		dim = len(input_shape)
		shape = [convShape(i, k, s, d, p, t) for i, k, s, d, p, t in zip(input_shape,
			listify(kernel, dim),
			listify(stride, dim),
			listify(dilation, dim),
			listify(padding, dim),
			listify(stride_transpose, dim))]
		return shape

	return (input_shape*stride_transpose + 2*padding - dilation*kernel - 1)//stride + 1

def autoShape(input_shape, kernel, output_shape, max_dilation=3, max_stride_transpose=4):
	if any([isinstance(p, Iterable) for p in [input_shape, kernel, output_shape]]):
		dim = len(input_shape)
		shape = [autoShape(i, k, o, max_dilation, max_stride_transpose) for i, k, o in zip(input_shape,
			listify(kernel, dim),
			listify(output_shape, dim))]
		stride_transpose, padding, dilation, stride = np.transpose(shape)
		stride, dilation, stride_transpose = stride + 1, dilation + 1, stride_transpose + 1
		return stride.tolist(), dilation.tolist(), padding.tolist(), stride_transpose.tolist()

	predictions = np.array([[[[convShape(input_shape, kernel, s, d, p, t)
		for s in range(1, kernel*d + 1)] + [-1]*((max_dilation - d)*kernel)
		for d in range(1, max_dilation + 1)]
		for p in range(kernel//2 + 1)]
		for t in range(1, max_stride_transpose + 1)])

	cost = predictions - output_shape
	cost[cost < 0] = np.amax(cost) + 1
	return list(np.unravel_index(np.argmin(cost), cost.shape))

"""
padding functions
"""

def pad(x, padding, padding_mode='constant', padding_value=0):
    padding = listify(padding, x.ndim)
    padding = np.repeat(padding[::-1], 2)
    return F.pad(input=x, pad=tuple(padding), mode=padding_mode, value=padding_value)

def Pad(padding, mode='constant', value=0):
	return lambda input: pad(input, padding, mode, value)

"""
slicing functions
"""

def view(x, kernel, stride=1, dilation=1):
    strided, ndim = x.clone(), x.ndim
    kernel, stride, dilation= [listify(p, ndim) for p in [kernel, stride, dilation]]
    for dim, (k, s, d) in enumerate(zip(kernel, stride, dilation)):
        strided = strided.unfold(dim, k*d, s)
        if d != 1:
            idx = torch.LongTensor(range(k*d)[::d])
            strided = torch.index_select(strided, -1, idx)
            
    return strided

def View(kernel, stride=1):
	return lambda input: view(input, kernel, stride)

"""
custom layers
"""

class Flatten(nn.Module):
	def forward(self, input):
		return input.flatten(1)

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, input):
		shape = listify(self.shape, input.ndim)
		return input.reshape(-1, *shape)
