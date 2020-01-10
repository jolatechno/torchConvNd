from .utils.utils import *

"""
n-D convolutional network with automatic output shape
"""

def convNdRec(x, hidden, shapes, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, *args):
	out, hidden_out = x.clone(), np.zeros(0)
	idx = 0
	for shape in shapes:
		stride, dilation, padding, stride_transpose = autoShape(list(out.shape), kernel, shape, max_dilation, max_stride_transpose)

		filled = x.clone()
		for dim, s in enumerate(listify(stride_transpose, x.ndim)):
			filled = filled.repeat_interleave(s, dim)

		padded = pad(filled, padding, padding_mode, padding_value)
		strided = view(padded, kernel, stride, dilation)
		inter, batch_shape = strided.flatten(0, x.ndim - 1), strided.shape[:x.ndim]

		batch_size = inter.shape[0]

		if idx + batch_size > hidden.shape[0]:
			dif = idx + batch_size - hidden.shape[0]
			hidden = torch.cat((hidden, torch.zeros(dif, *hidden.shape[1:])), 0)

		result, hidden[idx:idx + batch_size] = func(inter.unsqueeze(1), hidden.narrow(0, idx, batch_size), *args)

		idx += batch_size

		out = result.reshape(*batch_shape)
		for dim, s in enumerate(listify(shape, x.ndim)):
			if out.shape[dim] != s:
				out = out.narrow(dim, 0, s)
		
	return out, hidden

class ConvNdRec(nn.Module):
	def __init__(self, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4):
		super(ConvNdRec, self).__init__()
		self.parameters = func.parameters
		self.forward = lambda x, hidden, shapes, *args: convNdRec(x, hidden, shapes, func, kernel, padding_mode, padding_value, max_dilation, max_stride_transpose, *args)