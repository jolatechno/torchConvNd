from network import *
from datasets import *

from matplotlib import pyplot as plt

trainX, trainY = load_mnist(False)

Net = network([5, 5], 70, [28, 28], 10, 4, [[50, 30, 10], [70, 70]], True)

hidden = [torch.rand(10, 0, 70) for i in range(4)]

out, hidden = Net(trainX.narrow(0, 0, 10).unsqueeze(1), hidden)

print(out)
print([h.shape for h in hidden])