import torch
import matplotlib.pyplot as plt
from torch import nn

# 1. Create data
# Create the data parameters
weight = 0.3
bias = 0.9
# Make X and y using linear regression feature
X = torch.arange(0,100,0.1).unsqueeze(dim = 1)
y = weight * X + bias
print(X.dtype)


# 2. Split the data into training and testing
train_split = int(len(X) * 0.8)
X_train = X[:train_split]
y_train = y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]

# 3. Plot the training and testing data 
def plot_predictions(train_data,train_labels,test_data,test_labels,predictions=None):

    "Plots training data, test data dn compares predictions"

    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_labels, c="k", s=6, label = "training data")
    plt.scatter(test_data, test_labels, c="g", s=6, label = "Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=6, label= "Predictions")
    
    plt.legend(prop={"size": 14})
    plt.show()

# plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test)

# 4. Make a NN model
class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features = 1,
                                  out_features = 1)
  def forward(self,x : torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)
  
torch.manual_seed(42)
model_1 = LinearRegressionModel()
print(model_1.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_1.parameters(),
                            lr = 0.0001)

epochs = 1000

# 5. Train the model
for epoch in range(epochs):
  
  model_1.train()
  y_pred = model_1(X_train)
  loss = loss_fn(y_pred,y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  with torch.inference_mode():
    if epoch % 100 == 0:
      model_1.eval()
      y_preds = model_1(X_test)
      test_loss = loss_fn(y_preds,y_test)
      print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test loss: {test_loss:.3f}")


# 6. Make predictions with the model
model_1.eval()

with torch.inference_mode():
   y_preds = model_1(X_test)

plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=y_preds)
