import pandas as pd
import numpy as np
import torch


def load_fer2013(df=None, cfg=None):
    """
    按照 FER2013 官方 Usage 标签严格加载并划分数据集。

    官方划分规则:
      - 'Training'   -> 纯训练集 (28,709张)
      - 'PublicTest' -> 验证集 (3,589张) -> 用作训练时的 val_loss 观察
      - 'PrivateTest'-> 独立测试集 (3,589张) -> 暂时归入验证，或你可以单独返回
    """
    # 1. 读取原始 CSV
    df = pd.read_csv(cfg["load_path"])

    # 2. 提取像素矩阵
    print("[load_fer2013] 正在解析像素矩阵，请稍候...")
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    x = np.vstack(pixels.values).reshape(-1, 48, 48)  # (总样本数, 48, 48)
    y = df['emotion'].values

    # 3. 核心：根据官方 Usage 列建立布尔掩码 (Mask)
    # 如果你的 CSV 对应的列名是 'Usage'，直接按下面切分
    train_mask = (df['Usage'] == 'Training').values

    # 划重点：官方有 PublicTest 和 PrivateTest。
    # 为了榨干所有验证数据的价值，我们可以把 PublicTest 作为实时验证集 (val)
    val_mask = (df['Usage'] == 'PublicTest').values

    # 如果你想把 PrivateTest 也一起合进验证集里（扩大验证规模），可以解除下面这行的注释：
    # val_mask = (df['Usage'] == 'PublicTest').values | (df['Usage'] == 'PrivateTest').values

    # 4. 依照掩码分流数据
    x_train_raw = x[train_mask]
    y_train_raw = y[train_mask]

    x_val_raw = x[val_mask]
    y_val_raw = y[val_mask]

    # 5. 转换为 PyTorch Tensor (这里保持 0~255 即可，因为你的新 Preprocessing 内部会自动除以 255.0)
    # 增加通道维度变为 (N, 1, 48, 48)
    x_train = torch.tensor(x_train_raw).unsqueeze(1)
    y_train = torch.tensor(y_train_raw, dtype=torch.long)

    x_val = torch.tensor(x_val_raw).unsqueeze(1)
    y_val = torch.tensor(y_val_raw, dtype=torch.long)

    print(f"[load_fer2013] 官方标准划分加载完毕:")
    print(f"  -> 训练集样本数: {x_train.shape[0]} (期望: 28709)")
    print(f"  -> 验证集样本数: {x_val.shape[0]} (PublicTest 期望: 3589)")

    return x_train, x_val, y_train, y_val, "0"