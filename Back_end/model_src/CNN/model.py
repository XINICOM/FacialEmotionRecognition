import torch
import torch.nn as nn

default_config = {
        "features": [
                        # 格式: ["conv", out, k, pad, stroke, act, bn_lr_factor, drop2d, conv_lr_factor]
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


# ---------- 定义模型结构 ----------
def get_activation(act_name):
    if not act_name: return None
    name = act_name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    return None


class IndependentLrDynamicNet(nn.Module):
    def __init__(self, config=default_config, num_classes=7, input_channels=1, input_size=(48, 48)):
        super().__init__()

        self.param_lr_groups = []
        feature_layers = []
        current_channels = input_channels

        for layer_cfg in config.get("features", []):
            layer_type = layer_cfg[0]

            if layer_type == "conv":
                # 解构全配置项
                out_channels = layer_cfg[1]
                kernel_size = layer_cfg[2]
                padding = layer_cfg[3] if len(layer_cfg) > 3 else 0
                stride = layer_cfg[4] if len(layer_cfg) > 4 else 1
                act_name = layer_cfg[5] if len(layer_cfg) > 5 else "relu"

                # BN 的特殊处理：可以是 False，也可以是独立的学习率倍数数字
                bn_val = layer_cfg[6] if len(layer_cfg) > 6 else False
                drop_2d = layer_cfg[7] if len(layer_cfg) > 7 else 0.0
                conv_lr_factor = layer_cfg[8] if len(layer_cfg) > 8 else 1.0

                # 1. 卷积层添加与参数绑定
                conv_layer = nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride, padding=padding)
                feature_layers.append(conv_layer)
                # 显式命名，方便我们辨认
                self.param_lr_groups.append({
                    "params": conv_layer.parameters(),
                    "lr_factor": conv_lr_factor,
                    "desc": f"Conv2d({current_channels}->{out_channels})"
                })

                # 2. BatchNorm 层添加与【完全独立】参数绑定
                # 只要 bn_val 不是 False 且不是 0，就开启 BN
                if bn_val is not False and bn_val != 0:
                    # 如果传的是 True，默认给 1.0 倍率；如果传的是数字，就用该数字作为 BN 的独立倍率
                    bn_lr_factor = bn_val if isinstance(bn_val, (int, float)) else 1.0

                    bn_layer = nn.BatchNorm2d(out_channels)
                    feature_layers.append(bn_layer)
                    self.param_lr_groups.append({
                        "params": bn_layer.parameters(),
                        "lr_factor": bn_lr_factor,  # 使用完全独立的 BN 学习率倍数
                        "desc": f"BatchNorm2d({out_channels})"
                    })

                # 3. 激活与 Dropout
                act_layer = get_activation(act_name)
                if act_layer: feature_layers.append(act_layer)
                if drop_2d > 0: feature_layers.append(nn.Dropout2d(p=drop_2d))

                current_channels = out_channels

            elif layer_type == "pool":
                pool_type = layer_cfg[1] if len(layer_cfg) > 1 else "max"
                kernel_size = layer_cfg[2] if len(layer_cfg) > 2 else 2
                stride = layer_cfg[3] if len(layer_cfg) > 3 else kernel_size
                if pool_type == "max":
                    feature_layers.append(nn.MaxPool2d(kernel_size, stride=stride))
                elif pool_type == "avg":
                    feature_layers.append(nn.AvgPool2d(kernel_size, stride=stride))

        self.features = nn.Sequential(*feature_layers)

        # 计算维度
        dummy_input = torch.zeros(1, input_channels, input_size[0], input_size[1])
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flattened_dim = dummy_output.numel()

        # 分类器层
        classifier_layers = [nn.Flatten()]
        prev_dim = flattened_dim

        for idx, layer_info in enumerate(config.get("classifier", [])):
            h_dim = layer_info[0]
            act_name = layer_info[1] if len(layer_info) > 1 else "relu"
            drop_p = layer_info[2] if len(layer_info) > 2 else 0.0
            linear_lr_factor = layer_info[3] if len(layer_info) > 3 else 1.0

            linear_layer = nn.Linear(prev_dim, h_dim)
            classifier_layers.append(linear_layer)
            self.param_lr_groups.append({
                "params": linear_layer.parameters(),
                "lr_factor": linear_lr_factor,
                "desc": f"Linear_Hidden_{idx}({prev_dim}->{h_dim})"
            })

            act_layer = get_activation(act_name)
            if act_layer: classifier_layers.append(act_layer)
            if drop_p > 0: classifier_layers.append(nn.Dropout(p=drop_p))
            prev_dim = h_dim

        final_layer = nn.Linear(prev_dim, num_classes)
        classifier_layers.append(final_layer)
        self.param_lr_groups.append({
            "params": final_layer.parameters(),
            "lr_factor": 1.0,
            "desc": f"Linear_Output({prev_dim}->{num_classes})"
        })

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_optimizer_groups(self, base_lr):
        optimizer_groups = []
        for group in self.param_lr_groups:
            optimizer_groups.append({
                "params": group["params"],
                "lr": base_lr * group["lr_factor"],
                "desc": group["desc"]  # 顺带把描述传过去，方便调试打印
            })
        return optimizer_groups