import torch

# ---------------------- CNN ----------------------
from Back_end.model_src.CNN.train_model import train_CNN
from Back_end.model_src.CNN.model import EmotionCNN
# ------------------ other_model ------------------


def train(cfg, train_loader, val_loader):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = cfg["model_type"]
    model_path = cfg["load_model_path"]

    if model_type == "CNN":
        model = EmotionCNN(num_classes=7)
        if model_path != "0":
            # 加载参数（state_dict 会加载到 CPU 内存）
            state_dict = torch.load(model_path, map_location='cpu')  # 强制在 CPU 上加载
            model.load_state_dict(state_dict)

        return train_CNN(model, train_loader, val_loader, epochs=cfg["epochs"],
                         device=device, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                         save_path=cfg["save_path"], verbose=cfg["verbose"])

    # elif model_type == "RNN":
    #     train_RNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
