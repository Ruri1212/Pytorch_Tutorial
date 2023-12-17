import torch
import torch.onnx as onnx
import torchvision.models as models


### modelの重みを保存する
model = models.vgg16(pretrained = True)
torch.save(model.state_dict(),"model_weight.pth")
model = models.vgg16()
model.load_state_dict("model_weight.pth")

## dropoutやbatch_normを推論モードにする
model.eval()



### modelを保存する
### 独自クラスを使用する場合，先に宣言しておかないとloadでエラーになる
torch.save(model,"path")
model = torch.load("path")
