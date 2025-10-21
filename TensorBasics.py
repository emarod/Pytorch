import torch

# BASIC OPERATIONS

# Addition (Element Wise)
print("******* Adition ********\n")
x = torch.rand(2)
y = torch.rand(2)

print("Tensor x:", x)
print("Tensor y:", y)

z = x + y
print(z)
z = torch.add(x, y)
print(z)
print(x.add_(y))

# Substraction (Element Wise)
print("\n******* Substraction ********\n")
x = torch.rand(2)
y = torch.rand(2)

print("Tensor x:", x)
print("Tensor y:", y)

z = x - y
print(z)
z = torch.sub(x, y)
print(z)
print(x.sub_(y))

# Multiplication
print("\n******* Multiplication ********\n")
x = torch.rand(2)
y = torch.rand(2)

print("Tensor x:", x)
print("Tensor y:", y)

z = x * y
print(z)
z = torch.mul(x, y)
print(z)
print(x.mul_(y))

# Division
print("\n******* Division ********\n")
x = torch.rand(2)
y = torch.rand(2)

print("Tensor x:", x)
print("Tensor y:", y)

z = x / y
print(z)
z = torch.div(x, y)
print(z)
print(x.div_(y))
print(x)
