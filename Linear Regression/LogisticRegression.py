import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# print("X_train", X_train.shape, X_train)
# print("X_test", X_test.shape, X_test)
# print("y_train", y_train.shape, y_train)
# print("y_test", y_test.shape, y_test)

y_train = y_train.view(y_train.shape[0], 1) # Column vector
y_test = y_test.view(y_test.shape[0], 1)

# print(y_train)
# print(y_test)

# Model
n_samples, n_features = X_train.shape

class LogisticRegression(nn.Module):

  def __init__(self, n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1) #feature vector dim, output dim

  def forward(self, x):
    y_predicted = torch.sigmoid(self.linear(x))
    return y_predicted

model = LogisticRegression(n_features)

# Loss and optimizer
criterion = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
num_epochs = 1000
for epoch in range(num_epochs):
  # forward pass and loss
  y_predicted = model.forward(X_train)
  loss = criterion(y_predicted, y_train)
  # backward pass
  # f(X) = sigmoid(WX + b)
  # W, b
  loss.backward()
  # updates
  #w = w - learning_rate*dw
  optimizer.step()
  optimizer.zero_grad()

  if (epoch+1) % 10 == 0:
    print(f"Epoch {epoch + 1}, loss = {loss.item():.4f}")

with torch.no_grad():
  y_predicted = model(X_test)
  y_predicted_classes = y_predicted.round()
  acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
  print(f"Accuracy = {acc: .4f}")
