import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms


class EnhancedFERDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        # 此时传入的 x_tensor 必须已经是 0~1 的 FloatTensor，形状为 (N, 1, 48, 48)
        self.x = x_tensor
        self.y = y_tensor.long()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 极简切片，不包含任何循环、最大值查找、类型转换和除法！
        img = self.x[idx]
        label = self.y[idx]

        # 仅做最轻量级的标准化
        return self.normalize(img), label


def preprocessing(preprocessing_cfg, x_train, x_val, y_train, y_val):
    batch_size = preprocessing_cfg.get("batch_size", 128)

    print("[preprocessing] 正在一次性将全量数据做预缩放 (0~1)...")

    # 【核心改动】：在外面一次性解决全量数据的归一化和维度确保
    x_train = x_train.float()
    if x_train.max() > 1.0:
        x_train = x_train / 255.0
    if x_train.dim() == 3:  # (N, 48, 48) -> (N, 1, 48, 48)
        x_train = x_train.unsqueeze(1)

    x_val = x_val.float()
    if x_val.max() > 1.0:
        x_val = x_val / 255.0
    if x_val.dim() == 3:
        x_val = x_val.unsqueeze(1)

    # 此时装配出来的 Dataset，其内部的 __getitem__ 极其干净
    train_dataset = EnhancedFERDataset(x_train, y_train)
    val_dataset = EnhancedFERDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0  # 因为此时 __getitem__ 已经没有任何计算量了，0 就能跑出飞一般的速度
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    print(f"[preprocessing] 数据管道组装完毕")
    return train_loader, val_loader, "0"