from .utils.utils import *
from .convNdFunc import convNdFunc

"""
n-D convolutional network with automatic output shape
"""

def convNdAutoFunc(x, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, *args):
	stride, dilation, padding, stride_transpose = autoShape(list(x.shape)[1:], kernel, shape, max_dilation)
	return convNdFunc(x, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)

class ConvNdAutoFunc(nn.Module):
	def __init__(self, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3):
		super(ConvNdAutoFunc, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, *args: convNdAutoFunc(x, shape, func, kernel, padding_mode, padding_value, max_dilation, *args)

"""
n-D convolutional network with automatic output shape and linear filter
"""

def convNdAuto(x, weight, shape, kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3):
	def func(x):
		flat = x.flatten()
		return nn.functional.linear(flat, weight, bias)

	return convNdAutoFunc(x, shape, func, kernel, padding_mode, padding_value, max_dilation)

class ConvNdAuto(nn.Module):
	def __init__(self, shape, kernel, bias=False, padding_mode='constant', padding_value=0, max_dilation=3):
		super(ConvNdAuto, self).__init__()
		model = nn.Sequential(
			Flatten(),
			nn.Linear(np.prod(kernel), 1, bias))

		conv = ConvNdAutoFunc(shape, model, kernel, padding_mode, padding_value, max_dilation)
		self.parameters = conv.parameters
		self.forward = conv.forward
