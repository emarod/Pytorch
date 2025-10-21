import torch


x = torch.tensor(1, dtype=torch.float32)
y = torch.tensor(2, dtype=torch.float32)
w = torch.tensor(1, requires_grad=True, dtype=torch.float32)

# Forward Pass
y_hat = w * x
s = y_hat - y
l = s * s

# Backward pass
l.backward()
print(w.grad)

# weights = torch.ones(4, requires_grad=True) 

# for epoch in range(5):
#   model_output = (weights * 3).sum()
#   model_output.backward()
#   print(weights.grad)
#   weights.grad.zero_()