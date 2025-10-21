import torch

X = torch.tensor([1,2,3,4,5], dtype=torch.float32)
y = torch.tensor([2,4,6,8,10], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
  return w * x

def loss(y, y_pred):
  return ((y-y_pred)**2).mean()

# Training
lr = 0.01
n_iters = 100

for epoch in range(n_iters):
  y_pred = forward(X)
  l = loss(y, y_pred)
  l.backward()
  #update weights
  with torch.no_grad():
    w -= lr*w.grad
  w.grad.zero_()

  if epoch % 10 == 0:
    print(f"Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")