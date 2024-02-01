import torch
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda

dataset = datasets.FashionMNIST(
    root="data",
    train = True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor[y],1))
)

### Lamadaはtransformとしてlambda関数を使うため)
### scatterの変更位置指定(中間のargument) はtensor型
a = torch.zeros(10,dtype = torch.float)
b = torch.tensor([3])
c = a.scatter(0,b,1)
print(c)
print(a)

