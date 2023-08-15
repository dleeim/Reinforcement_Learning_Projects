import torch
from torch import nn

# load data
data = []

# define NN model
class model_softmax(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features = 1,
                                  out_features = 1)
    
  def forward(self,x : torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)

# define loss, optimizier
optimizer = torch.optim.SGD(model_softmax.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()

# train the NN model
Loss = []
epochs = 100
for epoch in range(epochs):

    for x, y in data:
        optimizer.zero_grad()
        y_pred = model_softmax(x)
        loss = criterion(y_pred, y)
        Loss.append(loss)
        loss.backward()
        optimizer.step()

print("Done!")