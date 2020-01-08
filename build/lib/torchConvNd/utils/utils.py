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

	if isinstance(x, Iterable):
		if len(x) != dims:
			ValueError("Shape don't match up")

		return x

	return [x for i in range(dims)]

def sequencify(x, nlist=1, dims=1):
	if nlist < 0:
		return x
	if isinstance(x, Iterable):
		if isinstance(x[0], Iterable):
			return x
		return [x for i in range(nlist)]
	return [[x for i in range(dims)] for i in range(nlist)]

def extendedLen(x):
	if isinstance(x, Iterable):
		return len(x)
	return -1

def dimCheck(*args):
    dim = max([extendedLen(x)for x in args])
    return (*[listify(x, dim) for x in args], dim == -1)

"""
functions for convNdAuto
"""

def calcShape(input_shape, kernel, stride=1, dilation=1, padding=0):
	input_shape, kernel, stride, dilation, padding, single = dimCheck(input_shape, kernel, stride, dilation, padding)
	if single:
		return ((input_shape + 2*padding - dilation*(kernel - 1) - 1)//stride + 1)*dilation
	
	return [calcShape(i, k, s, d, p) for i, k, s, d, p in zip(input_shape, kernel, stride, dilation, padding)]

def autoStridePad(input_shape, output_shape, kernel, dilation=1):
	input_shape, output_shape, kernel, dilation, single = dimCheck(input_shape, output_shape, kernel, dilation)

	if single:
		if output_shape < calcShape(input_shape, kernel, kernel, dilation, 0):
			raise ValueError("output shape is too small to be reached in one layer (without loosing inforamtion)")

		if output_shape > calcShape(input_shape, kernel, 1, dilation, kernel//2):
			raise ValueError("output shape is too big to be reached in one layer")

		for stride in range(1, kernel + 1):
			if calcShape(input_shape, kernel, stride, dilation, 0) <= output_shape:
				break

		padding = 0
		while padding <= kernel//2:
			if calcShape(input_shape, kernel, stride, dilation, padding) >= output_shape:
				break

			if padding == kernel//2:
				stride -= 1
				padding = 0

			padding += 1

		stride, padding = 1, 1
		return stride, padding
	
	return [autoStridePad(i, o, k, d) for i, o, k, d in zip(input_shape, output_shape, kernel, dilation)]

def AutoStridePad(kernel, dilation=1):
	return lambda input_shape, output_shape: autoStridePad(input_shape, output_shape, kernel, dilation)

"""
padding functions
"""

def pad(input, padding, mode='constant', value=0):
	padding = listify(padding, input.ndim)
	padding = np.repeat(padding[::-1], 2)	
	return F.pad(input=input, pad=tuple(padding), mode=mode, value=value)

def Pad(padding, mode='constant', value=0):
	return lambda input: pad(input, padding, mode, value)

"""
slicing functions
"""

def view(input, kernel, stride=1):
    strided, kernel, stride = input, listify(kernel, input.ndim), listify(stride, input.ndim)
    
    for dim, (k, s) in enumerate(zip(kernel, stride)):
    	strided = strided.unfold(dim, k, s)

    return strided

def View(kernel, stride=1):
	return lambda input: view(input, kernel, stride)

"""
functions to prepare a convolution
"""

def convPrep(input, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0):
	kernel, stride, padding, _ = dimCheck(kernel, stride, padding)

	in_dim = input.ndim
	padded = pad(input, padding, padding_mode, padding_value)
	strided = view(input, kernel, stride)

	shape = strided.shape[:in_dim]
	return strided.flatten(0, in_dim - 1), shape

def ConvPrep(input, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0):
	Fpad = Pad(padding, padding_mode, padding_value)
	Fstride = View(kernel, stride)
	
	return lambda input: Fstride(Fpad(input))

"""
functions to postprocess a convolution result
"""

letters = 'abcdefghijklmnopqrstuvwxyz'

def convPost(input, shape):
	input = input.reshape(*shape, *input.shape[1:])

	dim = input.ndim//2
	command = letters[:2*dim] + " -> "
	for i in range(dim):
	    command = command + letters[i] + letters[i + dim]

	out = torch.einsum(command, input)
	for i in range(dim):
	    out = out.flatten(i, i + 1)

	return out

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
