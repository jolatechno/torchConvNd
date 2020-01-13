from .utils.utils import *
from .convNdFunc import convNdFunc 

"""
n-D recurent convolutional network
"""

def convNdRec(x, hidden, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
	batch_length, seq, hidden_shape = *x.shape[:2], hidden.shape[1:]
	x = x.permute(1, 0, *range(2, x.ndim)).flatten(0, 1)
	idx, hiddens = [0], [hidden] # Has to be a list to be modified inside of Func

	def Func(y, *args):
		y = y.reshape(seq, -1, *kernel).permute(1, 0, *range(2, y.ndim + 1))

		length = y.shape[0]
		dif = idx[0] + length - hiddens[0].shape[0]
		if dif > 0:
			fill = torch.zeros(dif, *hidden_shape)
			hiddens[0] = torch.cat((hiddens[0], fill), 0)
			
		out, hiddens[0][:, idx[0]:idx[0] + length] = func(y, hiddens[0][:, idx[0]: idx[0] + length], *args)
		out = out.permute(1, 0, *range(2, out.ndim)).flatten(0, 1)

		idx[0] += length
		return out
	
	result = convNdFunc(x, Func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)
	result = result.reshape(seq, batch_length, *result.shape[1:]).permute(1, 0, *range(2, result.ndim + 1))
	return result, hiddens[0]

class ConvNdRec(nn.Module):
	def __init__(self, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0):
		super(ConvNdRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, shape, *args: convNdRec(x, hidden, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)

"""
n-D recurent convolutional network with automatic output shape
"""

def convNdAutoRec(x, hidden, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False, *args):
	stride, dilation, padding, stride_transpose = autoShape(list(x.shape)[2:], kernel, shape, max_dilation, max_stride_transpose)
	out = convNdRec(x, hidden, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)
	if Clip:
		return clip(out, [-1] + listify(shape, x.ndim - 1))
	return out

class ConvNdAutoRec(nn.Module):
	def __init__(self, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False):
		super(ConvNdAutoRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, shape, *args: convNdAutoRec(x, hidden, shape, func, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose, Clip, *args)
