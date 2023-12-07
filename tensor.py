import numpy as np
import torch

#####  型変換  #####
# リストからtensor
data = [[1,2],[3,4]]
print(type(data))
data = torch.tensor(data)
print(type(data))
# numpyからtensor
data = np.array(data)
print(type(data))
data = torch.from_numpy(data)
print(type(data))
print()


#####  tensorの作成  #####
## torch.rangeは[]の範囲
## torch.arangeはpythonと同じ
# テンソルをshapeを使って作成
shape = (2,3)
data = torch.rand(shape)
print(data)
data = torch.ones(shape)
print(data)
data = torch.zeros(shape)
print(data)
print()


##### テンソルの属性変数 #####
# # テンソルの属性変数
print(data.shape)
print(data.dtype)
# # テンソルをGPU上に移動
# # print(torch.cuda.is_available())
# # if torch.cuda.is_available():
# #     print(data.device)
# #     data.to("cuda")
# #     print(data.device)
# #     data = data.to("cuda")
# #     print(data.device)
print()


##### テンソル表示 #####
data = torch.randn(3,4)
print(data)
print("first row",data[0])
print("first column",data[:,0])
print("last column",data[:,-1])
print()


##### テンソル結合 #####
## dim = 0 は縦に結合していく -> (12,3)
## dim = 1 は横に結合していく -> (4,9)
a = torch.arange(6).reshape(2,3)
b = torch.arange(8).reshape(2,4)
c = torch.cat([a,b],dim=1)
print(c)
b = torch.randn(2,3)
### dim = 0,1,2
c = torch.stack([a,b],dim=0)
print(c)
print()

##### テンソル算術 #####
a = torch.arange(6).reshape(2,3)
b = torch.arange(6).reshape(2,3)
## 行列の積
c = a @ b.T
print(c)
## 要素ごとの積
c = a * b
print(c)
print()



##### 1要素テンソル #####
a = torch.arange(6)
b = a.sum()
print(b)
c = b.item()
print(c,type(c))
print()

##### テンソルとNumpyの相互 #####
a = np.array([[1,2],[3,4]])
b = torch.from_numpy(a)
b.add_(1)
print(a)
print(b)
print()
a = torch.arange(4).reshape(2,2)
b = a.numpy()
a.add_(1)
print(a)
print(b)


##### テンソル結合 #####