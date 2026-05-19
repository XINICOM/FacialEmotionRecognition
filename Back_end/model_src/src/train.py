import torch
from torch.utils.data import DataLoader, TensorDataset


from Back_end.model_src.src.load_data import load_fer2013
# ---------------------- CNN ----------------------
from Back_end.model_src.CNN.train_model import train_CNN
from Back_end.model_src.CNN.model import EmotionCNN
# ------------------ other_model ------------------



def train(cfg, train_loader, val_loader):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = cfg["model_type"]
    if model_type == "CNN":
        model = EmotionCNN(num_classes=7)
        return train_CNN(model, train_loader, val_loader, epochs=cfg["epochs"],
                         device=device, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                         save_path=cfg["save_path"], verbose=cfg["verbose"])

    # elif model_type == "RNN":
    #     train_RNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


#  def train_CNN(model_src, train_loader, val_loader, epochs, device,
#                 lr=0.001, weight_decay=0.0, save_path="best_model.pth", verbose=True):