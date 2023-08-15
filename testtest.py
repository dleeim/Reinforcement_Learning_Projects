import torch
from torch import nn

m = nn.LogSoftmax(dim=0)
a = torch.Tensor([1,2])
print(m(a))
