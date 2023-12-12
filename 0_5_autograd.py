import torch

x = torch.ones(3)
y = torch.zeros(2)

w = torch.randn(2,3,requires_grad=True)
b = torch.randn(2,requires_grad=True)

z = torch.matmul(w,x) + b

print(z.grad_fn)
print(b.grad_fn)
print()
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
print(w.grad)
print(b.grad)
loss.backward()
print(w.grad)
print(b.grad)
print()
with torch.no_grad():
    z = torch.matmul(w,x) + b
print(z.requires_grad)