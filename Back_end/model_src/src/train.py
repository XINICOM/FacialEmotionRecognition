import torch
import json

# ---------------------- CNN ----------------------
from Back_end.model_src.CNN.train_model import train_CNN
from Back_end.model_src.CNN.train_model import train_CNN_console

from Back_end.model_src.CNN.model import IndependentLrDynamicNet
# ------------------ other_model ------------------

empty_model = model_layers = {"features":[],"classifier":[]}
# default_model = {
#         "features":   [
#             ["conv", 32, 3, 1, 1, "relu", 0.2, 0.1, 0.5],
#             ["pool", "max", 2, 2],
#             ["degradable_conv", 64, 3, 1, 1, "leaky_relu", 1.0, 0.2, 1.2],
#             ["pool", "avg", 4, 4],
#             ["degradable_conv", 128, 3, 1, 1, "gelu", 0.0, 0.3, 2.0],
#         ],
#         "classifier": [
#             [256, "leaky_relu", 0.5, 1.0],
#             [128, "relu", 0.3, 1.5],
#             [32 , "relu", 0.3, 1.5]
#         ]
#     }

# default_model = {
#     "features": [
#         # 阶段 1：基础特征提取（纯 Conv），死死锁住空间结构
#         ["conv", 32, 3, 1, 1, "relu", 0.2, 0.1, 0.5],
#         ["pool", "max", 2, 2],  # 48x48 -> 24x24
#
#         # 阶段 2：中层特征提取（纯 Conv），继续提取五官几何
#         ["conv", 64, 3, 1, 1, "leaky_relu", 1.0, 0.2, 1.0],
#         ["pool", "max", 2, 2],  # 24x24 -> 12x12
#
#         # 阶段 3：高层特征提取（纯 Conv），把通道数顶上去
#         ["conv", 128, 3, 1, 1, "gelu", 1.0, 0.2, 1.0],
#         ["pool", "max", 2, 2],  # 12x12 -> 6x6
#
#         # 阶段 4：临门一脚（唯一的动态可退化层！）
#         # 此时特征图是 6x6，尺寸小、参数轻量（128*6*6*128），绝配！
#         # 最后的 2.5 倍学习率倍数给得非常精准，能推着全连接快速收敛
#         ["degradable_conv", 128, 3, 1, 1, "gelu", 0.0, 0.3, 2.5],
#     ],
#     "classifier": [
#         # 因为最后一层退化层已经完成了“全图跨域联动”，后面的分类器可以大幅度瘦身
#         [128, "leaky_relu", 0.5, 1.5],
#         [32, "relu", 0.5, 1.5]
#     ]
# }

default_model = {
    "features": [
        ["conv", 32, 3, 1, 1, "relu", 0.2, 0.1, 0.5],
        ["pool", "max", 2, 2],

        # 适当提高中间卷积层的 Dropout (从 0.2 调到 0.3)，在特征源头注入噪声
        ["conv", 64, 3, 1, 1, "leaky_relu", 0.3, 0.2, 1.0],
        ["pool", "max", 2, 2],

        # 高层卷积也加上 0.3 的 Dropout
        ["conv", 128, 3, 1, 1, "gelu", 0.3, 0.2, 1.0],
        ["pool", "max", 2, 2],

        # 可退化层保持 0.0（因为 FC 路径本身需要稳定的参数更新，我们在分类器里统一拦截它）
        ["degradable_conv", 128, 3, 1, 1, "gelu", 0.0, 0.3, 2.5],
    ],
    "classifier": [
        # 【手术核心】：砍掉 32 那一层，直接 256 -> 128 -> 7。
        # 每一层 FC 后面，全部死死按住 0.5 的高强度 Dropout！
        [256, "leaky_relu", 0.5, 1.0],
        [128, "relu", 0.5, 1.5],  # 这里的 Dropout 从 0.3 暴力提升到 0.5
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
