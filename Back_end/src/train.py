"""
训练模块 - 模型训练与评估，支持 pause / resume / terminate
"""
import torch
import torch.nn as nn
import torch.optim as optim
from flask import jsonify

import config as cfg
import control


def train(model, train_loader, val_loader, start_epoch=0, best_val_acc=0.0,
          optimizer_state=None, scheduler_state=None):
    """
    训练模型
    支持从指定epoch和最优精度恢复训练
    """
    try:
        model = model.to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)

        patience_counter = 0

        print(f"[train] 开始训练 | 设备: {cfg.DEVICE} | 总轮数: {cfg.NUM_EPOCHS}")
        print(f"[train] 起始轮次: {start_epoch} | 历史最优验证精度: {best_val_acc:.4f}")

        for epoch in range(start_epoch, cfg.NUM_EPOCHS):
            # 检查控制信号
            control.check_pause()
            if control.check_terminate():
                print(f"[train] 训练被用户终止于 epoch {epoch}")
                break

            # ---- 训练阶段 ----
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} "
                          f"| Loss: {loss.item():.4f}")

            train_acc = train_correct / train_total
            avg_train_loss = train_loss / train_total

            # ---- 验证阶段 ----
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            scheduler.step(val_acc)

            yield jsonify({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "total_epoch": cfg.NUM_EPOCHS})

            # ---- 保存最优模型 ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                }, cfg.MODEL_SAVE_PATH)
                print(f"[train] 最优模型已保存 (验证精度: {best_val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                    print(f"[train] 早停触发 (连续 {cfg.EARLY_STOP_PATIENCE} 轮验证精度未提升)")
                    break

        print(f"[train] 训练结束 | 最优验证精度: {best_val_acc:.4f}")
        
        successful = "0"
    except Exception as e:
        successful = str(e)
    yield jsonify({"successful": successful})


def evaluate(model, data_loader, criterion=None):
    """评估模型在给定数据集上的表现"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total if criterion else 0.0
    accuracy = correct / total
    return avg_loss, accuracy


def test(model, test_loader):
    """在测试集上评估模型并输出分类报告"""
    model.eval()
    model = model.to(cfg.DEVICE)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(cfg.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    print(f"\n[test] 测试集总精度: {correct}/{total} = {correct/total:.4f}")
    print("[test] 各类别精度:")

    for cls_id in range(len(cfg.EMOTION_LABELS)):
        cls_total = sum(1 for l in all_labels if l == cls_id)
        cls_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == cls_id and p == cls_id)
        if cls_total > 0:
            print(f"  {cfg.EMOTION_LABELS[cls_id]}: {cls_correct}/{cls_total} = {cls_correct/cls_total:.4f}")
