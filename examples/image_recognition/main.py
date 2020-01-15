from network import *
from datasets import *

from matplotlib import pyplot as plt

Net = network([5, 5], 70, [28, 28], 10, 4, [[50, 30, 10], [70, 70]], True)

tensor = torch.rand(1, 1, 28, 28)
hidden = [torch.rand(0, 70) for i in range(4)]
cost = torch.rand(1, 1)

out, hidden = Net(tensor, hidden, cost)

print(out)
print([h.shape for h in hidden])