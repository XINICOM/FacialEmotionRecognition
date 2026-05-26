import torch.optim as optim
import torchvision.transforms.v2 as transforms
import time

from Back_end.shared_state import get_controller
from Back_end.model_src.CNN.optimizer import configure_optimizer_and_scheduler
from Back_end.model_src.CNN.alpha import calculate_annealing_alpha


def train_CNN(model, train_loader, val_loader, epochs, device,
                lr=0.001, weight_decay=0.0, save_path="best_model.pth", verbose=True):
    """
    :param verbose: 布尔值。True 则打印每个 Epoch 的日志；False 则完全静音。
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    best_val_acc = 0.0

    ctrl = get_controller()
    for epoch in range(epochs):
        if ctrl.wait_if_paused_or_terminated():
            print("收到终止信号，退出任务")
            return history


        # ==================== 训练阶段 ====================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(running_loss)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)


        # ==================== 保存最佳模型 ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch

        # ==================== 控制台打印控制 ====================
        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] -> "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% || "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        # ====================== 流式传输 =======================
        yield {"Epoch": epoch + 1,
               "train_loss": avg_train_loss, "train_acc": train_acc,
               "val_loss": avg_val_loss, "val_acc": val_acc}

    if verbose:
        print("train正常完成")
    yield "0"
    return None




# def train_CNN_console(model, train_loader, val_loader, epochs, device,
#                 lr=0.001, weight_decay=0.0, save_path="best_model.pth", verbose=True):
#     """
#     :param verbose: 布尔值。True 则打印每个 Epoch 的日志；False 则完全静音。
#     """
#     model = model.to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     history = {
#         'train_loss': [],
#         'train_acc': [],
#         'val_loss': [],
#         'val_acc': [],
#         'best_val_acc': 0.0,
#         'best_epoch': 0
#     }
#     best_val_acc = 0.0
#
#     for epoch in range(epochs):
#         # ==================== 训练阶段 ====================
#         model.train()
#         running_loss = 0.0
#         correct_train = 0
#         total_train = 0
#
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#
#             _, predicted = torch.max(outputs, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()
#
#         avg_train_loss = running_loss / len(train_loader)
#         train_acc = correct_train / total_train
#
#         history['train_loss'].append(avg_train_loss)
#         history['train_acc'].append(train_acc)
#
#         # ==================== 验证阶段 ====================
#         model.eval()
#         val_loss = 0.0
#         correct_val = 0
#         total_val = 0
#
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total_val += labels.size(0)
#                 correct_val += (predicted == labels).sum().item()
#
#         avg_val_loss = val_loss / len(val_loader)
#         val_acc = correct_val / total_val
#
#         history['val_loss'].append(avg_val_loss)
#         history['val_acc'].append(val_acc)
#
#         # ==================== 控制台打印控制 ====================
#         if verbose:
#             print(f"Epoch [{epoch + 1}/{epochs}] -> "
#                   f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% || "
#                   f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
#
#         # ==================== 保存最佳模型 ====================
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), save_path)
#             history['best_val_acc'] = best_val_acc
#             history['best_epoch'] = epoch
#
#     return history


import torch
import torch.nn as nn


def train_CNN_console(model, train_loader, val_loader, epochs, device,
                      lr=0.001, weight_decay=0.0, save_path="best_model.pth", verbose=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    gpu_transform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    )

    # ==================== 1. 瘦身成果：一句话配置好优化器和调度器 ====================
    optimizer, scheduler = configure_optimizer_and_scheduler(model, lr, weight_decay, epochs)

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'best_val_acc': 0.0, 'best_epoch': 0
    }
    best_val_acc = 0.0

    for epoch in range(epochs):

        # ==================== 2. 瘦身成果：更新可退化层的 Alpha 开关 ====================
        alpha_val = calculate_annealing_alpha(epoch, epochs)
        model.update_alpha(alpha_val)

        # ==================== 训练阶段 ====================
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for k, (images, labels) in enumerate(train_loader):
            t0 = time.time()

            images = images.to(device)
            labels = labels.to(device)
            t1 = time.time()  # 挪动数据耗时

            inputs = gpu_transform(images)
            t2 = time.time()  # 空间增强耗时

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t3 = time.time()  # GPU 计算耗时

            if k % 50 == 0:
                print(
                    f"Batch {k} -> 数据搬运: {t1 - t0:.4f}s | 空间增强: {t2 - t1:.4f}s | 神经网络计算: {t3 - t2:.4f}s")

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # ==================== 3. 动态学习率步进（为后续缝合做的完美埋伏） ====================
        # 获取当前的学习率（用于日志打印展示）
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            # ⚡ 自动化判断：如果是看指标的 ReduceLROnPlateau，必须喂入验证集准确率
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)  # 喂入你的验证集准确率
            else:
                # 如果是普通的 CosineAnnealingLR、StepLR 等静态调度器，直接无脑步进
                scheduler.step()

            # ==================== 控制台打印控制 ====================
        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] (Alpha: {alpha_val:.3f} | LR: {current_lr:.6f}) -> "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% || "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        # ==================== 保存最佳模型 ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch

    return history