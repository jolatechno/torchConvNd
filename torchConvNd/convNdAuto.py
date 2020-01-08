from .utils.utils import *
from .convNdFunc import convNdFunc, ConvNdFunc

from torch import nn

"""
n-D convolutional layer with automatic output shape
"""

def convNdAuto(input, func, kernel, dilation=1, padding_mode='constant', padding_value=0, *args):
	pass

class ConvNdAuto(nn.Module):
	def __init__(self, func, kernel, dilation=1, padding_mode='constant', padding_value=0):
		self.parameters = func.parameters
		self.forward = lambda input, output_shape, *args: (input, func, kernel, dilation, padding_mode, padding_value, *args)
