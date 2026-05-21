import torch

# ---------------------- CNN ----------------------
from Back_end.model_src.CNN.predict_model import predict_CNN
from Back_end.model_src.CNN.model import IndependentLrDynamicNet
# ------------------ other_model ------------------



def predict(cfg):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = cfg['load_model_path']

    model_type = cfg["model_type"]
    if model_type == "CNN":
        model = IndependentLrDynamicNet(num_classes=7)
        # 加载参数（state_dict 会加载到 CPU 内存）
        state_dict = torch.load(model_path, map_location='cpu')  # 强制在 CPU 上加载
        model.load_state_dict(state_dict)

        return predict_CNN(model, img_path=cfg["load_img_path"], device=device)

    # elif model_type == "RNN":
    #     train_RNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


