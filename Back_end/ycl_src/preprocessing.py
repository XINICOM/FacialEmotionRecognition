"""
数据预处理模块 - 数据增强、标准化、DataLoader 构建
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import IMG_SIZE, BATCH_SIZE, NUM_CLASSES


class FERDataset(Dataset):
    """人脸表情数据集"""

    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        img = self.pixels[idx].reshape(IMG_SIZE, IMG_SIZE).astype(np.uint8)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(img).unsqueeze(0) / 255.0

        label = torch.tensor(label, dtype=torch.long)
        return img, label


def get_train_transform():
    """训练集数据增强"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transform():
    """验证/测试集预处理"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def preprocessing(train_pixels, train_labels, val_pixels, val_labels,
                  test_pixels, test_labels):
    """
    构建训练、验证、测试 DataLoader
    返回: (train_loader, val_loader, test_loader)
    """
    train_dataset = FERDataset(train_pixels, train_labels, get_train_transform())
    val_dataset = FERDataset(val_pixels, val_labels, get_eval_transform())
    test_dataset = FERDataset(test_pixels, test_labels, get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"[preprocessing] DataLoader 构建完成")
    print(f"  训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}, 测试批次数: {len(test_loader)}")
    return train_loader, val_loader, test_loader
