import torch
from torch import nn
a = torch.Tensor([1,2,3,4,5])
b = nn.functional.softmax(a)
print(b)