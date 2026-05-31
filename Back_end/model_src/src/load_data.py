import pandas as pd
import numpy as np
import torch


def load_fer2013(cfg=None):
    """
    按照 FER2013 官方 Usage 标签划分数据集

    官方划分规则:
      - 'Training'   -> 训练集 (28,709张)
      - 'PublicTest' -> 验证集 (3,589张)
      - 'PrivateTest'-> 独立测试集 (3,589张)
    """
    # 1. 读取 CSV
    if cfg is None:
        df = pd.read_csv("D:\\pycharm\\homework\\FacialEmotionRecognition\\Back_end\\data\\train\\fer2013.csv")
    else:
        df = pd.read_csv(cfg["load_path"].replace(",,", "/"))

    # 2. 提取像素矩阵
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    x = np.vstack(pixels.values).reshape(-1, 48, 48)  # (总样本数, 48, 48)
    y = df['emotion'].values

    # 3. 根据官方 Usage 列建立布尔掩码 (Mask)
    # Training 作为训练集 (train)
    train_mask = (df['Usage'] == 'Training').values

    # PublicTest 作为实时验证集 (val)
    val_mask = (df['Usage'] == 'PublicTest').values

    # PrivateTest 作为最终测试集 (test)
    test_mask = (df['Usage'] == 'PrivateTest').values

    # 4. 依照掩码分流数据
    x_train_raw = x[train_mask]
    y_train_raw = y[train_mask]

    x_val_raw = x[val_mask]
    y_val_raw = y[val_mask]

    x_test_raw = x[test_mask]
    y_test_raw = y[test_mask]

    # 5. 转换为 PyTorch Tensor (在 Preprocessing 内部除以 255.0)
    # 增加通道维度变为 (N, 1, 48, 48)
    x_train = torch.tensor(x_train_raw).unsqueeze(1)
    y_train = torch.tensor(y_train_raw, dtype=torch.long)

    x_val = torch.tensor(x_val_raw).unsqueeze(1)
    y_val = torch.tensor(y_val_raw, dtype=torch.long)

    x_test = torch.tensor(x_test_raw).unsqueeze(1)
    y_test = torch.tensor(y_test_raw, dtype=torch.long)

    return x_train, x_val, y_train, y_val, x_test, y_test