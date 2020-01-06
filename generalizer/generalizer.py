import utils

from torch import nn

class Generalizer(nn.Modules):
	def __init__(self, net, in_shape, out_shape):
		self.net = net
		self.parameters = net.parameters

		self.in_shape, self.out_shape = in_shape, out_shape
		
	def forward(self, input, cost=0):
		pass

	def reset(self, length):
		self.params = self
