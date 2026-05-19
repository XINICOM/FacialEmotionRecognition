import torch
from torch import nn
from torch import optim

import torch
import torch.nn as nn
import torch.optim as optim


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

    for epoch in range(epochs):
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

        # ==================== 控制台打印控制 ====================
        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] -> "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% || "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        # ==================== 接口扩展点（可选） ====================
        # 如果你以后想做训练进度的实时推流（比如前端有个进度条），可以在这里把数据包装成 json 发送出去：
        # current_status = {"epoch": epoch+1, "train_loss": avg_train_loss, "val_acc": val_acc}
        # send_to_websocket(current_status)

        # ==================== 保存最佳模型 ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch

    return history




