from .utils.utils import *
from .convNdFunc import convNdFunc

"""
n-D convolutional network with automatic output shape
"""

def convNdAutoFunc(x, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, clip=False, *args):
	stride, dilation, padding, stride_transpose = autoShape(list(out.shape), kernel, shape, max_dilation, max_stride_transpose)
	out = convNdFunc(x, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)
	
	if clip:
		for dim, s in enumerate(listify(shape, x.ndim)):
			if out.shape[dim] != s:
				out = out.narrow(dim, 0, s)
	return out

class ConvNdAutoFunc(nn.Module):
	def __init__(self, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
		super(ConvNdAutoFunc, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, shapes, *args: convNdAutoFunc(x, shapes, func, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose, *args)

"""
n-D convolutional network with automatic output shape and linear filter
"""

def convNdAuto(x, weight, shapes, kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
	def func(x):
		flat = x.flatten()
		return nn.functional.linear(flat, weight, bias)

	return convNdAutoFunc(x, shapes, func, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose)

class ConvNdAuto(nn.Module):
	def __init__(self, kernel, bias=False, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
		super(ConvNdAuto, self).__init__()
		model = nn.Sequential(
			Flatten(),
			nn.Linear(np.prod(kernel), 1, bias))

		conv = ConvNdAutoFunc(model, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose)
		self.parameters = conv.parameters
		self.forward = conv.forward
