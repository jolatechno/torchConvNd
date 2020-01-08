from .utils.utils import *
from .convNdFunc import convNdFunc, ConvNdFunc

from torch import nn

"""
n-D convolutional layer with automatic output shape
"""

def convNdAuto(input, output_shape, func, kernel, dilation=1, padding_mode='constant', padding_value=0, *args):
	stride, padding = autoStridePad(list(input.shape), output_shape, kernel, dilation)
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
