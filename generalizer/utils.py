import torch

def Stride(kernel, stride=1, padding=None):
	pad_length = padding if padding is not None else kernel//(stride*2)
	def separate(input):
		padded = torch.cat((torch.zeros(pad_length),
							input,
							torch.zeros(pad_length)), 0).unsqueeze(0)
		strided = torch.cat([input.narrow(1, i*stride, kernel)
							 for i in range(len(padded)//stride)], 0)
		return strided
	return separate