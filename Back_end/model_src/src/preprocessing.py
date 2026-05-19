import torch
from torch.utils.data import DataLoader, TensorDataset

def preprocessing(preprocessing_cfg,x_train,x_val,y_train,y_val):

    batch_size = preprocessing_cfg["batch_size"]
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # 原来的第六行
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader