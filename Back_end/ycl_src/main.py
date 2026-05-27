"""
主入口模块 - 人脸情感识别系统
提供交互式菜单，集成 load_data / preprocessing / train / predict / pause / resume / terminate
"""
import os
import torch
from config import (DEVICE, MODEL_SAVE_PATH, EMOTION_LABELS)
from load_data import load_data
from preprocessing import preprocessing
from model import EmotionCNN
from train import train, test
from predict import predict_image
import control


def menu():
    """显示交互式菜单"""
    print("\n" + "=" * 55)
    print("       人脸情感识别系统 (FER-CNN)")
    print("=" * 55)
    print("  1. 加载数据并预处理 (load_data + preprocessing)")
    print("  2. 开始训练 (train)")
    print("  3. 恢复训练 (resume)")
    print("  4. 暂停训练 (pause)")
    print("  5. 终止训练 (terminate)")
    print("  6. 测试模型 (test)")
    print("  7. 预测单张图片 (predict)")
    print("  8. 一键运行: 加载+训练+测试 (full pipeline)")
    print("  0. 退出")
    print("=" * 55)


def do_load_data():
    """加载数据并构建 DataLoader"""
    data = load_data()
    train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels = data
    train_loader, val_loader, test_loader = preprocessing(
        train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels
    )
    return train_loader, val_loader, test_loader


def do_train(train_loader, val_loader, resume_from_checkpoint=False):
    """执行训练"""
    model = EmotionCNN()
    start_epoch = 0
    best_val_acc = 0.0
    optimizer_state = None
    scheduler_state = None

    if resume_from_checkpoint and os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        print(f"[main] 从 checkpoint 恢复: epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")
    else:
        control.clear_all_flags()

    model, best_val_acc = train(
        model, train_loader, val_loader,
        start_epoch=start_epoch, best_val_acc=best_val_acc,
        optimizer_state=optimizer_state, scheduler_state=scheduler_state
    )
    return model


def do_test(model, test_loader):
    """执行测试"""
    test(model, test_loader)


def do_predict():
    """交互式预测"""
    image_path = input("请输入图片路径: ").strip().strip('"').strip("'")
    if image_path:
        predict_image(image_path)
    else:
        print("[main] 图片路径不能为空")


def do_full_pipeline():
    """一键运行完整流程"""
    print("[main] ===== 一键运行: 加载数据 -> 训练 -> 测试 =====")
    train_loader, val_loader, test_loader = do_load_data()
    model = do_train(train_loader, val_loader, resume_from_checkpoint=False)
    do_test(model, test_loader)
    print("[main] ===== 完整流程结束 =====")


def main():
    """主循环"""
    train_loader, val_loader, test_loader = None, None, None
    model = None

    print(f"[main] 设备: {DEVICE}")
    print(f"[main] 情感类别: {list(EMOTION_LABELS.values())}")

    while True:
        menu()
        choice = input("请选择操作 [0-8]: ").strip()

        if choice == "1":
            train_loader, val_loader, test_loader = do_load_data()

        elif choice == "2":
            if train_loader is None:
                print("[main] 请先加载数据 (选项1)")
                continue
            model = do_train(train_loader, val_loader, resume_from_checkpoint=False)

        elif choice == "3":
            if train_loader is None:
                print("[main] 请先加载数据 (选项1)")
                continue
            model = do_train(train_loader, val_loader, resume_from_checkpoint=True)

        elif choice == "4":
            control.pause()

        elif choice == "5":
            control.terminate()

        elif choice == "6":
            if test_loader is None:
                print("[main] 请先加载数据 (选项1)")
                continue
            if model is None and not os.path.exists(MODEL_SAVE_PATH):
                print("[main] 请先训练模型 (选项2)")
                continue
            if model is None:
                model = EmotionCNN()
                checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"])
            do_test(model, test_loader)

        elif choice == "7":
            do_predict()

        elif choice == "8":
            do_full_pipeline()
            # 更新变量以供后续操作使用
            train_loader_tmp, val_loader_tmp, test_loader_tmp = do_load_data()
            train_loader, val_loader, test_loader = train_loader_tmp, val_loader_tmp, test_loader_tmp

        elif choice == "0":
            print("[main] 退出系统")
            control.clear_all_flags()
            break

        else:
            print("[main] 无效选择，请重新输入")


if __name__ == "__main__":
    main()
