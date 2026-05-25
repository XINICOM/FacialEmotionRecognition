import torch
import json

# ---------------------- CNN ----------------------
from Back_end.model_src.CNN.train_model import train_CNN
from Back_end.model_src.CNN.train_model import train_CNN_console

from Back_end.model_src.CNN.model import IndependentLrDynamicNet
# ------------------ other_model ------------------

empty_model = model_layers = {"features":[],"classifier":[]}
default_model = {
        "features":   [
            ["conv", 32, 3, 1, 1, "relu", 0.2, 0.1, 0.5],
            ["pool", "max", 2, 2],
            ["conv", 64, 3, 1, 1, "leaky_relu", 1.0, 0.2, 1.2],
            ["pool", "avg", 4, 4],
            ["conv", 128, 3, 1, 1, "gelu", 0.0, 0.3, 2.0],
        ],
        "classifier": [
            [256, "leaky_relu", 0.5, 1.0],
            [128, "relu", 0.3, 1.5],
            [32 , "relu", 0.3, 1.5]
        ]
    }

def train_stream_packer(cfg, train_loader, val_loader):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = cfg["model_type"]

    model_path = cfg["load_model_path"]

    if model_type == "CNN":
        model_layers = cfg["model_layers"]
        if model_layers == empty_model:
            model_layers = default_model

        model = IndependentLrDynamicNet(num_classes=7,config=model_layers)
        if model_path != "0":
            # 加载参数（state_dict 会加载到 CPU 内存）
            state_dict = torch.load(model_path, map_location='cpu')  # 强制在 CPU 上加载
            model.load_state_dict(state_dict)
        try:
            # 调用核心训练函数，拿到训练生成器
            training_generator = train_CNN(model=model, train_loader=train_loader, val_loader=val_loader,
                                           epochs=cfg["epochs"],
                                           device=device, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                           save_path=cfg["save_path"], verbose=cfg["verbose"])
            # 循环读取并转发
            for metrics in training_generator:
                # 转换为标准前端流格式：'data: {"epoch": 1, ...}\n\n'
                yield f"data: {json.dumps(metrics)}\n\n"

            # 训练正常结束
            yield f"data: {json.dumps({'status': 'completed', 'message': '训练成功！'})}\n\n"

        except Exception as e:
            # 捕获整个训练周期的异常（如显存溢出、路径错误）并返回给前端
            yield f"data: {json.dumps({'status': 'failed', 'error': str(e)})}\n\n"

    # elif model_type == "RNN":
    #     train_RNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")




def train_stream_packer_console(cfg, train_loader, val_loader):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = cfg["model_type"]
    model_path = cfg["load_model_path"]

    if model_type == "CNN":
        # model_layers = cfg["model_layers"]
        # if model_layers == empty_model:
        model_layers = default_model
        model = IndependentLrDynamicNet(num_classes=7, config=model_layers)
        if model_path != "0":
            # 加载参数（state_dict 会加载到 CPU 内存）
            state_dict = torch.load(model_path, map_location='cpu')  # 强制在 CPU 上加载
            model.load_state_dict(state_dict)

        return train_CNN_console(model, train_loader, val_loader, epochs=cfg["epochs"],
                         device=device, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                         save_path=cfg["save_path"], verbose=cfg["verbose"])

    # elif model_type == "RNN":
    #     train_RNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
