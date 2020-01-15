import torch
from torch import nn

import numpy as np

from torchConvNd import ConvNdRec
from torchConvNd.utils import Clip, Flatten, autoShape

class cell(nn.Module):
	def __init__(self, kernel, hidden_length, hidden_shapes=[[], []], bias=False):
		super(cell, self).__init__()

		input_shape = np.prod(kernel) + hidden_length + 1
		i2o_shape = [input_shape] + hidden_shapes[0] + [1]
		i2h_shape = [input_shape] + hidden_shapes[1] + [hidden_length]

		self.i2o = nn.Sequential(*[nn.Linear(In, Out, bias) for In, Out in zip(i2o_shape[:-1], i2o_shape[1:])])
		self.i2h = nn.Sequential(*[nn.Linear(In, Out, bias) for In, Out in zip(i2h_shape[:-1], i2h_shape[1:])])

	def forward(self, x, hidden, cost):
		Out, In = [], x.flatten(2, -1)
		n_repeat = x.shape[0]//cost.shape[0]
		for i in range(x.shape[1]):
			Slice, Cost = In[:, i, :], cost[:, i]
			Cost = cost.repeat_interleave(n_repeat, 0)
			inter = torch.cat((Slice, hidden, Cost), -1)
			hidden = self.i2h(inter)
			Out.append(self.i2o(inter).squeeze(-1).unsqueeze(1))
		return torch.cat(Out, 1), hidden

class network(nn.Module):
	def __init__(self, kernel, hidden_length, starting_shape, n_feature, n_layers=1, hidden_shapes=[[], []], bias=False):
		super(network, self).__init__()
		func = cell(kernel, hidden_length, hidden_shapes, bias)

		final_shape = [n_feature, 1]
		shapes = [[(starting_shape[0]*(n_layers - i) + final_shape[0]*i)//n_layers,
			(starting_shape[1]*(n_layers - i) + final_shape[1]*i)//n_layers] for i in range(n_layers + 1)]
		params = [autoShape(In, kernel, Out) for In, Out in zip(shapes[:-1], shapes[1:])]

		self.layers = [ConvNdRec(func, *param) for param in params] + [Clip([-1, -1, n_feature, 1])]

		self.parameters = func.parameters

	def forward(self, x, hidden, cost):
		out = x.clone()
		for i, layer in enumerate(self.layers[:-1]):
			out, hidden[i] = layer(out, hidden[i], cost)
		clipped = self.layers[-1](out)
		return out.squeeze(-1), hidden