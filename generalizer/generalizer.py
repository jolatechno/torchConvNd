from utils import Stride

from torch import nn

class Generalizer(nn.Modules):
	def __init__(self, net, in_shape, out_shape):
		self.net = net
		self.parameters = net.parameters

		self.out_shape, self.in_shape = out_shape, in_shape if isinstance(in_shape, list) else [in_shape, in_shape]
		self.Istride, self.Pstride = Stride(self.in_shape[0], out_shape), Stride(self.in_shape[1], out_shape)
		
	def forward(self, input, cost=0):
		strided_input = Istride(input)
		strided_params = Pstride(self.params)
		expanded_cost = torch.Tensor([cost]*strided_input.shape[0]).unsqueeze(1)

		inter = cat((strided_input, strided_params, expanded_cost), 1)
		out = self.net(inter)

		result = out.narrow(1, 0, self.in_shape[0]).flatten()
		self.params = out.narrow(1, self.in_shape[0], self.in_shape[1]).flatten()

		return result


	def reset(self, length):
		self.params = torch.zeros(length//in_shape[0]*in_shape[1])
