from network import *
from datasets import *

from matplotlib import pyplot as plt

from torch.optim import Adam as optim



batch_size = 15
n_iter = 5



trainX, trainY = load_mnist(False)
trainX = trainX.unsqueeze(1).unsqueeze(1)

Net = network([5, 5], 70, [28, 28], 10, 4, [[50, 30, 10], [70, 70]], True)

hidden = [torch.rand(1, 0, 70) for i in range(4)]

cost = None

Otpim = optim(Net.parameters(), lr=0.01)

for i in range(n_iter):
	Cost = 0
	Otpim.zero_grad()
	for j in range(batch_size):
		idx = np.random.randint(0, trainX.shape[0])
		out, hidden = Net(trainX[idx], hidden, cost)
		cost = costFunc(out, trainY[idx])

		Cost += cost[0, 0]
	Cost.backward()
	Otpim.step()

	hidden = [h.detach() for h in hidden]
	print(Cost.detach()/batch_size)
