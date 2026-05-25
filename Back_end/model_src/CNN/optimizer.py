import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 示例：后续可以换成任意调度器


def configure_optimizer_and_scheduler(model, base_lr, weight_decay, total_epochs):
    """
    配置核心：将模型参数分组、初始化优化器，并预留动态学习率调度器接口
    """
    # 1. 提取支持分层学习率倍数 (lr_factor) 的专属参数组
    optimizer_groups = model.get_optimizer_groups(base_lr=base_lr)

    # 2. 初始化核心优化器
    optimizer = optim.Adam(optimizer_groups, lr=base_lr, weight_decay=weight_decay)

    # 3. 为后续做动态学习率预留的调度器 (Scheduler)
    # 默认实例化一个余弦退火调度器作为骨架
    # 作用是让 base_lr 随着 epoch 自动递减

    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    # scheduler = None
    return optimizer, scheduler

