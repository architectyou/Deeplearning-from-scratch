import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.tensor(1.0, requires_grad=True)
y = (x-3) * (x-6) * (x-4)

# 미분함수 구하기?
y.backward()

print(f"x_grad : {x.grad}")