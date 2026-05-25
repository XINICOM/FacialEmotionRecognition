import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------ 辅助激活函数 -------------
def get_activation(act_name):
    if not act_name: return nn.Identity()
    name = act_name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    return nn.Identity()


# ---------- 2. 标准卷积子模块 ----------
class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1,
                 act="relu", bn_val=False, drop_2d=0.0, lr_factor=1.0):
        super().__init__()
        self.param_groups = []

        # 完美的自动化判断：如果有 BN 自动设为 False；没有 BN 则保留 True
        has_bn = (bn_val is not False and bn_val != 0)

        # 自动控制 bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=not has_bn)

        self.param_groups.append({
            "params": self.conv.parameters(),
            "lr_factor": lr_factor,
            "desc": f"Conv2d({in_channels}->{out_channels})"
        })

        # BN 层
        if has_bn:
            bn_lr_factor = bn_val if isinstance(bn_val, (int, float)) else 1.0
            self.bn = nn.BatchNorm2d(out_channels)
            self.param_groups.append({
                "params": self.bn.parameters(),
                "lr_factor": bn_lr_factor,
                "desc": f"BatchNorm2d({out_channels})"
            })
        else:
            self.bn = nn.Identity()

        self.act = get_activation(act)
        self.drop = nn.Dropout2d(p=drop_2d) if drop_2d > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


# ---------- 3. 可退化卷积子模块 (新增) ----------
class DegradableConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, padding=0, stride=1, act="relu", bn_val=False, drop_2d=0.0, lr_factor=1.0,
                 feat_size=(48, 48)):
        super().__init__()
        self.param_groups = []
        self.H, self.W = feat_size

        # 路径 A: 局部卷积
        self.local_conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False)
        self.param_groups.append({"params": self.local_conv.parameters(), "lr_factor": lr_factor,
                                  "desc": f"Degradable_LocalConv({in_c}->{out_c})"})

        # 路径 B: 全局全连接 (用 1x1 卷积接收全图展平)
        self.global_fc = nn.Conv2d(in_c * self.H * self.W, out_c, kernel_size=1, bias=False)
        self.param_groups.append({"params": self.global_fc.parameters(), "lr_factor": lr_factor,
                                  "desc": f"Degradable_GlobalFC({in_c * self.H * self.W}->{out_c})"})

        # 双路径独立 BN 处理
        if bn_val is not False and bn_val != 0:
            bn_lr_factor = bn_val if isinstance(bn_val, (int, float)) else 1.0
            self.local_bn = nn.BatchNorm2d(out_c)
            self.global_bn = nn.BatchNorm2d(out_c)
            self.param_groups.append({"params": self.local_bn.parameters(), "lr_factor": bn_lr_factor,
                                      "desc": f"Degradable_LocalBN({out_c})"})
            self.param_groups.append({"params": self.global_bn.parameters(), "lr_factor": bn_lr_factor,
                                      "desc": f"Degradable_GlobalBN({out_c})"})
        else:
            self.local_bn = nn.Identity()
            self.global_bn = nn.Identity()

        # 公共后处理
        self.act = get_activation(act)
        self.drop = nn.Dropout2d(p=drop_2d) if drop_2d > 0 else nn.Identity()

        # 退火因子
        self.register_buffer('alpha', torch.tensor(0.0))

    def set_alpha(self, alpha_value):
        self.alpha.copy_(torch.tensor(alpha_value))

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"特征图尺寸不匹配！期望 {self.H}x{self.W}, 实际 {H}x{W}"

        # 局部
        out_local = self.local_bn(self.local_conv(x))
        _, _, H_out, W_out = out_local.shape

        # 全局
        x_flat = x.view(B, -1, 1, 1)
        out_global = self.global_bn(self.global_fc(x_flat)).expand(-1, -1, H_out, W_out)

        # 融合与后处理
        out = (1.0 - self.alpha) * out_local + self.alpha * out_global
        return self.drop(self.act(out))


# ---------- 4. 池化子模块 ----------
class PoolingBlock(nn.Module):
    def __init__(self, pool_type="max", kernel=2, s=None):
        super().__init__()
        stride = s if s is not None else kernel
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel, stride=stride)
        else:
            self.pool = nn.AvgPool2d(kernel, stride=stride)

    def forward(self, x):
        return self.pool(x)


# ---------- 5. 全连接子模块 ----------
class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act="relu", drop_p=0.0, lr_factor=1.0, idx=0, is_output=False):
        super().__init__()
        self.param_groups = []

        self.linear = nn.Linear(in_dim, out_dim)
        desc = f"Linear_Output({in_dim}->{out_dim})" if is_output else f"Linear_Hidden_{idx}({in_dim}->{out_dim})"
        self.param_groups.append({"params": self.linear.parameters(), "lr_factor": lr_factor, "desc": desc})

        self.act = get_activation(act) if not is_output else nn.Identity()
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.linear(x)))


# ==================================================================
# ---------- 主模型壳子 (IndependentLrDynamicNet) ----------
# ==================================================================
class IndependentLrDynamicNet(nn.Module):
    def __init__(self, config, num_classes=7, input_channels=1, input_size=(48, 48)):
        super().__init__()

        self.param_lr_groups = []
        self.degradable_layers = []  # 用于给外部 train.py 方便地统一调节 alpha

        # ---------- 组装特征提取器 ----------
        feature_layers = []
        current_channels = input_channels
        current_size = input_size  # 实时追踪当前特征图大小 (H, W)

        for layer_cfg in config.get("features", []):
            layer_type = layer_cfg[0]

            if layer_type in ["conv", "degradable_conv"]:  # 支持配置里写新的组件名
                out_channels = layer_cfg[1]
                kernel_size = layer_cfg[2]
                padding = layer_cfg[3] if len(layer_cfg) > 3 else 0
                stride = layer_cfg[4] if len(layer_cfg) > 4 else 1
                act_name = layer_cfg[5] if len(layer_cfg) > 5 else "relu"
                bn_val = layer_cfg[6] if len(layer_cfg) > 6 else False
                drop_2d = layer_cfg[7] if len(layer_cfg) > 7 else 0.0
                conv_lr_factor = layer_cfg[8] if len(layer_cfg) > 8 else 1.0

                if layer_type == "conv":
                    block = StandardConvBlock(current_channels, out_channels, kernel_size, padding, stride, act_name,
                                              bn_val, drop_2d, conv_lr_factor)
                else:
                    # 动态可退化层，必须传入当前准确的特征图尺寸 current_size
                    block = DegradableConvBlock(current_channels, out_channels, kernel_size, padding, stride, act_name,
                                                bn_val, drop_2d, conv_lr_factor, feat_size=current_size)
                    self.degradable_layers.append(block)

                feature_layers.append(block)
                self.param_lr_groups.extend(block.param_groups)  # 自动收集子模块的学习率组

                # 依据公式动态推导下一步的特征图大小，供下一层使用
                h_out = (current_size[0] + 2 * padding - kernel_size) // stride + 1
                w_out = (current_size[1] + 2 * padding - kernel_size) // stride + 1
                current_size = (h_out, w_out)
                current_channels = out_channels

            elif layer_type == "pool":
                pool_type = layer_cfg[1] if len(layer_cfg) > 1 else "max"
                kernel_size = layer_cfg[2] if len(layer_cfg) > 2 else 2
                stride = layer_cfg[3] if len(layer_cfg) > 3 else kernel_size

                block = PoolingBlock(pool_type, kernel_size, stride)
                feature_layers.append(block)

                # 动态更新池化后的特征图大小
                h_out = (current_size[0] - kernel_size) // stride + 1
                w_out = (current_size[1] - kernel_size) // stride + 1
                current_size = (h_out, w_out)

        self.features = nn.Sequential(*feature_layers)

        # 算扁平化维度
        flattened_dim = current_channels * current_size[0] * current_size[1]

        # ---------- 组装分类器 ----------
        classifier_layers = [nn.Flatten()]
        prev_dim = flattened_dim

        for idx, layer_info in enumerate(config.get("classifier", [])):
            h_dim = layer_info[0]
            act_name = layer_info[1] if len(layer_info) > 1 else "relu"
            drop_p = layer_info[2] if len(layer_info) > 2 else 0.0
            linear_lr_factor = layer_info[3] if len(layer_info) > 3 else 1.0

            block = LinearBlock(prev_dim, h_dim, act_name, drop_p, linear_lr_factor, idx=idx, is_output=False)
            classifier_layers.append(block)
            self.param_lr_groups.extend(block.param_groups)
            prev_dim = h_dim

        # 最后一层输出
        final_block = LinearBlock(prev_dim, num_classes, is_output=True)
        classifier_layers.append(final_block)
        self.param_lr_groups.extend(final_block.param_groups)

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        return self.classifier(self.features(x))

    def update_alpha(self, alpha_value):
        """提供一个一键统一修改模型内所有退化层 alpha 的接口"""
        for layer in self.degradable_layers:
            layer.set_alpha(alpha_value)

    def get_optimizer_groups(self, base_lr):
        optimizer_groups = []
        for group in self.param_lr_groups:
            optimizer_groups.append({
                "params": group["params"],
                "lr": base_lr * group["lr_factor"],
                "desc": group["desc"]
            })
        return optimizer_groups