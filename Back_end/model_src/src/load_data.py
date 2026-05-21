import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from Back_end.model_src.src.load_config_from_str import handle_str_config



def load_fer2013(df=None,cfg=None):
    """
    加载并预处理 FER2013 数据集，将其划分为训练集和验证集，并转换为 PyTorch Tensor。

    :param df: str, 转换后必须包含以下键:
                      - "test_size": float, 验证集比例 (例如 0.2)
                      - "random_state": int, 随机种子以确保结果可复现
                      - "stratify": bool, 是否进行分层抽样（保持类别比例平衡）
                      - "max_samples": int (可选), 限制读取的样本数，用于快速测试
    :return: x_train (Tensor): 训练集特征，形状为 (N_train, 1, 48, 48)，值域 0~1
             x_val (Tensor):   验证集特征，形状为 (N_val, 1, 48, 48)，值域 0~1
             y_train (Tensor): 训练集标签，形状为 (N_train,)，Long 类型
             y_val (Tensor):   验证集标签，形状为 (N_val,)，Long 类型
    """
    df = pd.read_csv(cfg["load_path"])
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    x = np.vstack(pixels.values).reshape(-1, 48, 48)  # (样本数, 48, 48)
    y = df['emotion'].values
    # 划分训练集和验证集 (80% / 20%)
    stratify_param = y if cfg.get("stratify", False) else None
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=stratify_param)
    # 转为 PyTorch Tensor，并增加通道维度 (N, 1, 48, 48)
    x_train = torch.tensor(x_train).unsqueeze(1) / 255.0
    x_val = torch.tensor(x_val).unsqueeze(1) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    return x_train, x_val, y_train, y_val, "0"
