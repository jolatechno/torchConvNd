from utils import *

from torch import nn
import numpy as np

"""
n-D convolution with arbitrary function
"""

def generalConvNd(input, func, kernel, stride=1, dilation=1, padding=0):
	pass

class GeneralConvNd(nn.Module):
	def __init__(self, func, kernel, stride=1, dilation=1, padding=0):
		super(GeneralConvNd, self).__init__()
		pass
		
	def forward(self, input):
		pass

"""
n-D convolution with a recursive function
"""

def recConvNd(input, mem, func, kernel, stride=1, dilation=1, padding=0):
	pass

class RecConvNd(nn.Module):
	def __init__(self, func, kernel,  stride=1, dilation=1, padding=0):
		super(RecConvNd, self).__init__()
		pass
		
	def forward(self, input, mem):
		pass