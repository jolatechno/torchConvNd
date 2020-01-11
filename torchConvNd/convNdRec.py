from .utils.utils import *
from .convNdAutoFunc import convNdFunc

"""
n-D recurent convolutional network
"""

def convNdRec(x, hidden, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args):
	class Func:
		idx = 0
		def __call__(self, x, *args):
			length = x.shape[0]
			dif = idx + length - hidden.shape[0]
			if dif > 0:
				hidden = torch.cat((hidden, torch.zeros(dif, hidden.shape[1:])), 0)
			
			out, hidden[idx:idx + length] = func(x, hidden[idx: idx + length], *args)
			idx += length
			return out
	
	result = convNdFunc(x, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)
	return result, hidden

class ConvNdRec(nn.Module):
	def __init__(self, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
		super(ConvNdRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, shapes, *arg: convNdRec(x, hidden, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)

"""
n-D recurent convolutional network with automatic output shape
"""

def convNdAutoRec(x, hidden, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False, *args):
	stride, dilation, padding, stride_transpose = autoShape(list(out.shape), kernel, shape, max_dilation, max_stride_transpose)
	out = convNdRec(x, hidden, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)
	if Clip:
		return clip(out, shape):
	return out

class ConvNdAutoRec(nn.Module):
	def __init__(self, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
		super(ConvNdAutoRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, shapes, *arg: convNdAutoRec(x, hidden, shape, func, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose, *args)
