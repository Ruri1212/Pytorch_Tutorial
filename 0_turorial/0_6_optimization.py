import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


# m = torch.nn.LogSoftmax(dim=1)
# input= torch.randn(3,5,requires_grad=True)
# print(input)
# print(m(input))s
# n = torch.nn.LogSoftmax(dim=0)
# print(n(input))


class CustomNeuralNetwork(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
     
    def forward(self,x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
    
def train_loop(datasets,model,loss_fn,optimizer):
    size = len(datasets)
    for batchs,(X,y) in enumerate(datasets):
        #予測と損失計算
        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.update()

def test_loop(datasets,model,loss_fn):
    size = len(datasets)
    test_loss = 0
    with torch.no_grad():
        for batchs,(X,y) in enumerate(datasets):
            pred = model(X)
            test_loss += loss_fn(pred,y)



training_data = datasets.FashionMNIST(
    root="data",
    train = True,
    download= True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root= "data",
    train=False,
    download= True,
    target_transform=ToTensor(),
)

batch_size = 64
epochs = 10
learning_rate = 0.01

train_dataloader = DataLoader(datasets,batch_size=batch_size)
test_dataloader = DataLoader(datasets,batch_size=batch_size)

model = CustomNeuralNetwork()        
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)


for t in range(epochs):
    train_loop()
    test_loop()





