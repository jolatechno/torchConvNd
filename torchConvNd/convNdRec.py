from .utils.utils import *

"""
n-D convolutional network with automatic output shape
"""

def convNdRec(x, hidden, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, *args):
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
	result = 
	return result, hidden

class ConvNdRec(nn.Module):
	def __init__(self, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
		super(ConvNdRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, shapes, *args: convNdRec(x, hidden, shapes, func, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose, *args)
