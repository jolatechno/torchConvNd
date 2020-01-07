import torch
from skimage.util.shape import view_as_windows as view, pad

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

def calcPadding(kernel, stride): #calculating the padding length such that the shape stays constant
	dim = max(extendedLen(kernel), extendedLen(stride)) #assumming that stride and dilation are equal
	kernel, stride = listify(kernel), listify(stride)

	if isinstance(kernel, list): #equvalent to isinstance(stride, list)
		return [k//2 - s//2 for k, s in zip(kernel, stride)]
	return kernel//2 - s//2

"""
padding functions
"""

def Pad(padding):
	return lambda input: pad(input, padding)

"""
slicing functions
"""

def View(kernel, stride=1):
	return lambda input: view(input, kernel, stride)

"""
functions to prepare a convolution
"""

def convPrep(input, kernel, stride=1, padding=0):
	padding = padding if padding is not None else calcPadding(kernel, stride)

	padded = pad(input, padding)
	strided = view(input, kernel, stride)

	return strided

def ConvPrep(input, kernel, stride=1, padding=0):
	padding = padding if padding is not None else calcPadding(kernel, stride)

	Fpad = Pad(padding)
	Fstride = View(kernel, stride)
	
	return lambda input: Fstride(Fpad(input))

"""
functions to postprocess a convolution result
"""

def convPost(input):
	padding = padding if padding is not None else calcPadding(kernel, dilation)

	padded = pad(input, padding)
	strided = stride(input, kernel, dilation)

	return strided

def ConvPost(input):
	padding = padding if padding is not None else calcPadding(kernel, dilation)

	Fpad = Pad(padding)
	Fstride = Stride(kernel, dilation)
	
	return lambda input: Fstride(Fpad(input))

"""
custom layers
"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten()

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

    def forward(self, input):
    	shape = listify(shape, input.ndim)
        return input.reshape(*shape)