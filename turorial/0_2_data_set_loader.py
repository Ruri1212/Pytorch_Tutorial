import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

labels_map= {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# im,_ = training_data[0]
# print(type(im))
# print(im.shape)
## 次元が(1)になっているものを削除する
# img = im.squeeze()
# print(img.shape)


### figureの使い方や.torch.randint.item()の使い方
# figure = plt.figure(figsize=(8,8))
# cols,rows = 2,2
# for i in range(1,cols*rows+1):
#     sample_index = torch.randint(len(training_data),size=(1,)).item()
#     img,label = training_data[sample_index]
#     figure.add_subplot(cols,rows,i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(),cmap="gray")
# plt.show()


### init,len,getitem の3つを定義する必要あり
# class CustomDataset(Dataset):
#     def __init__(self,):
#         super().__init__()        
#     def __len__(self):
#         pass
        
#     def __getitem__(self,idx):
#         image_path = os.path.join(self.imge_dir,self.image_data.iloc[idx,0])
#         image = read_image(image_path)
#         label = self.image_data.iloc[idx,1]
#         sample = {"image": image,"label":label }
#         pass


print(type(training_data))
train_dataloader = DataLoader(training_data,batch_size=4,shuffle=True)
print(type(train_dataloader))

### next,iterの使い方
## イテレータ = iter(リスト，配列)
## 　要素 = next(イテレータ)


train_features,train_labels = next(iter(train_dataloader))
print(train_features.numpy().shape)
print(train_labels.numpy().shape)


