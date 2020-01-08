from .utils.utils import *
from .convNdFunc import convNdFunc, ConvNdFunc

from torch import nn

"""
n-D convolutional layer with automatic output shape
"""

def convNdAuto(input, func, kernel, dilation=1):
	pass

class ConvNdAuto(nn.Module):
	def __init__(self, func, kernel, dilation=1):
		self.parameters = func.parameters
		self.forward = lambda input, output_shape, *args: (input, func, kernel, dilation, *args)
