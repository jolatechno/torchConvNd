import torch
from skimage.util.shape import view_as_windows as stride, pad

"""
helper functions
"""

def listify(x, dims=1):
	if isinstance(x, list) or dims < 0:
		return x
	return [x for i in range(dims)]

def extendedLen(x):
	if isinstance(x, list):
		return len(x)
	return -1

def calcPadding(kernel, dilation): #calculating the padding length such that the shape stays constant
	dim = max(extendedLen(kernel), extendedLen(dilation)) #assumming that stride and dilation are equal
	kernel, dilation = listify(kernel), listify(dilation)

	if isinstance(kernel, list): #equvalent to isinstance(dilation, list)
		return [k//2 - s//2 for k, s in zip(kernel, dilation)]
	return kernel//2 - s//2

"""
padding functions
"""

def Pad(padding):
	return lambda input: pad(input, padding)

"""
slicing functions
"""

def Stride(kernel, dilation=1):
	return lambda input: stride(input, kernel, dilation)

"""
functions to preper a convolution
"""

def convPrep(input, kernel, dilation=1, padding=0):
	padding = padding if padding is not None else calcPadding(kernel, dilation)

	padded = pad(input, padding)
	strided = stride(input, kernel, dilation)

	return strided

def ConvPrep(input, kernel, dilation=1, padding=0):
	padding = padding if padding is not None else calcPadding(kernel, dilation)

	Fpad = Pad(padding)
	Fstride = Stride(kernel, dilation)
	
	return lambda input: Fstride(Fpad(input))