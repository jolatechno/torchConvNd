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
