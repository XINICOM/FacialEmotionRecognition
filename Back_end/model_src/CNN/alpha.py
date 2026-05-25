import math


def calculate_annealing_alpha(current_epoch, total_epochs):
    """
    计算当前 Epoch 应该匹配的 Alpha 退火值
    策略：前 30% 纯卷积，中间 40% 渐进退化，最后 30% 纯全连接
    """
    start_decay_epoch = int(total_epochs * 0.3)
    end_decay_epoch = int(total_epochs * 0.7)

    if current_epoch < start_decay_epoch:
        return 0.0
    elif current_epoch > end_decay_epoch:
        return 0.5
    else:
        # 中间 40% 走顺滑的余弦退火过渡
        progress = (current_epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch)
        return 0.25 * (1.0 - math.cos(math.pi * progress))