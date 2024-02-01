import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary


print(torch.cuda.is_available())

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.liner_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()            
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.liner_relu_stack(x)
        return logits

model = NeuralNetwork().to("cuda")


# print(model)
# summary(model=model,input_size=(4,28,28))
X = torch.rand(2,28,28,device="cuda")
logits = model(X)
print(type(logits))
print(logits.shape)
y_probab = nn.Softmax(dim=1)(logits)
print(type(y_probab))
print(y_probab.shape)
# print(y_probab.argmax(0))
print(y_probab.argmax(1))
print()
print()

##### Networkのパラメータ確認，parameters()とstate_dict()
linear = nn.Linear(10,2)
print(list(linear.parameters()))
print(linear.named_parameters())
print(linear.state_dict())

