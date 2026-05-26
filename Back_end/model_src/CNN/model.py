import torch
import torch.nn as nn

# ------------ 辅助激活函数 -------------
def get_activation(act_name):
    if not act_name: return nn.Identity()
    name = act_name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True) # 加上 inplace 节约显存
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    return nn.Identity()


# ---------- 2. 标准卷积子模块 (已修正：内部 Dropout 默认关闭，由独立层控制) ----------
class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1,
                 act="relu", bn_val=False, drop_2d=0.0, lr_factor=1.0):
        super().__init__()
        self.param_groups = []

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
        # 向下兼容旧配置：如果传了 drop_2d，依然在内部做（不推荐），不传默认是 Identity
        self.drop = nn.Dropout2d(p=drop_2d) if drop_2d > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


# ---------- 3. 可退化卷积子模块 ----------
class DegradableConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, padding=0, stride=1, act="relu", bn_val=False, drop_2d=0.0, lr_factor=1.0,
                 feat_size=(48, 48)):
        super().__init__()
        self.param_groups = []
        self.H, self.W = feat_size

        self.local_conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False)
        self.param_groups.append({"params": self.local_conv.parameters(), "lr_factor": lr_factor,
                                  "desc": f"Degradable_LocalConv({in_c}->{out_c})"})

        self.global_fc = nn.Conv2d(in_c * self.H * self.W, out_c, kernel_size=1, bias=False)
        self.param_groups.append({"params": self.global_fc.parameters(), "lr_factor": lr_factor,
                                  "desc": f"Degradable_GlobalFC({in_c * self.H * self.W}->{out_c})"})

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

        self.act = get_activation(act)
        self.drop = nn.Dropout2d(p=drop_2d) if drop_2d > 0 else nn.Identity()
        self.register_buffer('alpha', torch.tensor(0.0))

    def set_alpha(self, alpha_value):
        self.alpha.copy_(torch.tensor(alpha_value))

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"特征图尺寸不匹配！期望 {self.H}x{self.W}, 实际 {H}x{W}"

        out_local = self.local_bn(self.local_conv(x))
        _, _, H_out, W_out = out_local.shape

        x_flat = x.view(B, -1, 1, 1)
        out_global = self.global_bn(self.global_fc(x_flat)).expand(-1, -1, H_out, W_out)

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


# ---------- 5. 全连接子模块 (升级版：完美支持 BatchNorm1d) ----------
class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act="relu", bn_val=False, drop_p=0.0, lr_factor=1.0, idx=0, is_output=False):
        super().__init__()
        self.param_groups = []

        has_bn = (bn_val is not False and bn_val != 0) and (not is_output)

        self.linear = nn.Linear(in_dim, out_dim, bias=not has_bn)
        desc = f"Linear_Output({in_dim}->{out_dim})" if is_output else f"Linear_Hidden_{idx}({in_dim}->{out_dim})"
        self.param_groups.append({"params": self.linear.parameters(), "lr_factor": lr_factor, "desc": desc})

        # 完美的 1D 批归一化注入
        if has_bn:
            bn_lr_factor = bn_val if isinstance(bn_val, (int, float)) else 1.0
            self.bn = nn.BatchNorm1d(out_dim)
            self.param_groups.append({
                "params": self.bn.parameters(),
                "lr_factor": bn_lr_factor,
                "desc": f"BatchNorm1d_Hidden_{idx}({out_dim})"
            })
        else:
            self.bn = nn.Identity()

        self.act = get_activation(act) if not is_output else nn.Identity()
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0 else nn.Identity()

    def forward(self, x):
        # 严格执行工业级标准顺序：Linear -> BN -> Act -> Dropout
        return self.drop(self.act(self.bn(self.linear(x))))


# ==================================================================
# ---------- 主模型壳子 (升级版，完美支持动态解析所有独立组件) ----------
# ==================================================================
class IndependentLrDynamicNet(nn.Module):
    def __init__(self, config, num_classes=7, input_channels=1, input_size=(48, 48)):
        super().__init__()

        self.param_lr_groups = []
        self.degradable_layers = []

        # ---------- 组装特征提取器 ----------
        feature_layers = []
        current_channels = input_channels
        current_size = input_size

        for layer_cfg in config.get("features", []):
            layer_type = layer_cfg[0]

            if layer_type in ["conv", "degradable_conv"]:
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
                    block = DegradableConvBlock(current_channels, out_channels, kernel_size, padding, stride, act_name,
                                                bn_val, drop_2d, conv_lr_factor, feat_size=current_size)
                    self.degradable_layers.append(block)

                feature_layers.append(block)
                self.param_lr_groups.extend(block.param_groups)

                # 动态推导特征图形状
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

                h_out = (current_size[0] - kernel_size) // stride + 1
                w_out = (current_size[1] - kernel_size) // stride + 1
                current_size = (h_out, w_out)

            # ✨ 新增：支持在 features 列表里塞独立的 Dropout / Dropout2d 算子
            elif layer_type in ["dropout2d", "dropout"]:
                p_val = layer_cfg[1] if len(layer_cfg) > 1 else 0.25
                if layer_type == "dropout2d":
                    feature_layers.append(nn.Dropout2d(p=p_val))
                else:
                    feature_layers.append(nn.Dropout(p=p_val))

        self.features = nn.Sequential(*feature_layers)

        # 算扁平化维度
        flattened_dim = current_channels * current_size[0] * current_size[1]

        # ---------- 组装分类器 ----------
        classifier_layers = [nn.Flatten()]
        prev_dim = flattened_dim

        for idx, layer_info in enumerate(config.get("classifier", [])):
            h_dim = layer_info[0]
            act_name = layer_info[1] if len(layer_info) > 1 else "relu"
            # ✨ 新增支持：第三个位置是 bn_val (True/False/1.0)，第四个位置是 drop_p，第五个是学习率系数
            bn_val = layer_info[2] if len(layer_info) > 2 else False
            drop_p = layer_info[3] if len(layer_info) > 3 else 0.0
            linear_lr_factor = layer_info[4] if len(layer_info) > 4 else 1.0

            block = LinearBlock(prev_dim, h_dim, act=act_name, bn_val=bn_val,
                                drop_p=drop_p, lr_factor=linear_lr_factor, idx=idx, is_output=False)
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