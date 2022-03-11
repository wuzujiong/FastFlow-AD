import logging
import os
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

class RandomRotation(T.RandomRotation):
    def __init__(self, p: float, degrees: int):
        super(RandomRotation, self).__init__(degrees)
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            fill = self.fill
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * F.get_image_num_channels(img)
                else:
                    fill = [float(f) for f in fill]
            angle = self.get_params(self.degrees)

            img = F.rotate(img, angle, self.resample, self.expand, self.center, fill)
        return img

class MVTecTrainDataset(ImageFolder):
    def __init__(self, root: str, cls: str):
        """
        We use random horizontal flip, vertical flip and rotation, with probabilities of 0.5, 0.3 and 0.7,
        respectively. It should be noted that some categories are not suitable for violent data augmentation.
        """
        if cls == 'cable' or cls == 'capsule' or cls == 'pill':
            transform = T.Compose([
                T.Resize(288),
                RandomRotation(0.7, degrees=5),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])
            ])
        elif cls == 'toothbrush' or cls == 'zipper' or cls == 'transistor':
            transform = T.Compose([
                T.Resize(288),
                T.RandomHorizontalFlip(0.5),
                RandomRotation(0.7, degrees=5),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])
            ])
        elif cls == 'metal_nut':
            transform = T.Compose([
                T.Resize(288),
                RandomRotation(0.7, 180),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])
            ])
        else:
            transform = T.Compose([
                T.Resize(288),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.3),
                RandomRotation(0.7, degrees=180),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])
            ])
        print(f"[{cls}] Transformations: {transform}")
        super(MVTecTrainDataset, self).__init__(
            root = os.path.join(root, cls, 'train'),
            transform = transform
        )

    def __getitem__(self, index: int):

        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_dataloader(self, num_workers = 8, batch_size = 32):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers)

class MVTecTestDataset(ImageFolder):
    def __init__(self, root: str, cls: str, transform = None, mask_transform = None):
        super(MVTecTestDataset, self).__init__(
            root = os.path.join(root, cls, 'test'),
            transform = T.Compose([
                T.Resize(272),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])
                ]),
            target_transform =  T.Compose([
                T.Resize(288),
                T.CenterCrop(256),
                T.ToTensor(),
            ])
        )

    def __getitem__(self, index):
        path, _ = self.samples[index]
        x = self.loader(path)

        if "good" in path:
            mask = Image.new('L', x.size)
            y = 0
        else:
            mask_path = path.replace("test", "ground_truth")
            mask_path = mask_path.replace(".png", "_mask.png")
            mask = self.loader(mask_path)
            y = 1

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        assert x.shape[1:] == mask.shape[1:], "Width and Height os image and mask must be the same"
        return x, mask, y

    def get_dataloader(self):
        return DataLoader(self, batch_size=1)

