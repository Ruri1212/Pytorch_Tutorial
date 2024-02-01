import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 画像の並び方を揃えるために、すべての画像ファイルをロードしてソートします
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 画像とマスクをロードします
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 各色は異なるインスタンスに対応しているため、
        # なお、値が0になっているインスタンスは背景となります。
        
        mask = Image.open(mask_path)
        # マスクに対してはRGBに変換していない点に注意してください。
        
        # PIL 画像を numpy 配列に変換します
        mask = np.array(mask)
        # インスタンスは異なる色でエンコードされています
        obj_ids = np.unique(mask)
        # 最初のIDは背景なので削除します
        obj_ids = obj_ids[1:]

        # カラー・エンコードされたマスクを、True/Falseで表現されたマスクに変換します
        masks = mask == obj_ids[:, None, None]

        # 各マスクのバウンディングボックスの座標を取得します
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # すべてtorch.Tensorに変換します
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # クラスは今回は1つだけ（人物）です
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # すべてのインスタンスを、iscrowd=0と仮定します
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
mask = Image.open('data/PennFudanPed/PedMasks/FudanPed00001_mask.png')