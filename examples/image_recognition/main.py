from network import *
from datasets import *

from matplotlib import pyplot as plt

from torch.optim import Adam as optim
import torch

Net = network([10, 10], 30, [28, 28], 10, 4, [[40, 10], [40, 40]], False)
Net.func.train(200, 5, 400)

trainX, trainY = load_mnist(True)
trainX = trainX.unsqueeze(1).unsqueeze(1)



hidden = [torch.rand(1, 0, 30) for i in range(4)]

cost = None

Otpim = optim(Net.parameters(), lr=0.01)

with torch.no_grad():
	for i in range(1000):
		Cost = 0
		for j in range(30):
			idx = np.random.randint(0, trainX.shape[0])
			out, hidden = Net(trainX[idx], hidden, cost)
			cost = costFunc(out, trainY[idx].float())

			Cost += cost[0, 0]
		print(Cost/30)
