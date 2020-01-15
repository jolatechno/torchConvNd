from .utils.utils import *
from .convNdFunc import convNdFunc 

"""
n-D recurent convolutional network
"""

def convNdRec(x, hidden, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
	batch_length, seq, hidden_shape = *x.shape[:2], hidden.shape[1:]
	x = x.permute(1, 0, *range(2, x.ndim)).flatten(0, 1)

	def Func(y, hidden, *args):
		y = y.reshape(seq, -1, *kernel).permute(1, 0, *range(2, y.ndim + 1))

		dif = y.shape[0] - hidden.shape[0]
		if dif > 0:
			fill = torch.zeros(dif, *hidden_shape)
			hidden = torch.cat((hidden, fill), 0)
		
		out, hidden = func(y, hidden, *args)
		out = out.permute(1, 0, *range(2, out.ndim)).flatten(0, 1)
		return out, hidden
	
	result = convNdFunc(x, Func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, hidden, *args)
	result, hidden, additional = result[0], result[1], result[2:]
	result = result.reshape(seq, batch_length, *result.shape[1:]).permute(1, 0, *range(2, result.ndim + 1))
	return (result, hidden) + additional

class ConvNdRec(nn.Module):
	def __init__(self, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0):
		super(ConvNdRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, *args: convNdRec(x, hidden, func, kernel, stride, dilation, padding, stride_transpose, padding_mode, padding_value, *args)
