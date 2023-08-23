import torch 
from torch import nn

m2 = nn.Linear(3,4)
print(m2.state_dict())