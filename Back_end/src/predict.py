"""
预测模块 - 单张图片情感预测
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from . import config as cfg
from .model import EmotionCNN


def predict_image(image_path):
    """
    对单张图片进行情感预测
    参数:
        image_path: 图片路径
        model_path: 模型权重路径 (默认使用最优模型)
    返回:
        预测的情感类别名称和各类别概率
    """
    model_path = cfg.MODEL_SAVE_PATH

    if not os.path.exists(image_path):
        print(f"[predict] 错误: 图片不存在 - {image_path}")
        return None, None

    if not os.path.exists(model_path):
        print(f"[predict] 错误: 模型文件不存在 - {model_path}")
        print("[predict] 请先训练模型")
        return None, None

    # 加载模型
    model = EmotionCNN(num_classes=7)
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.DEVICE)
    model.eval()

    # 预处理图片
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(cfg.DEVICE)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    emotion = cfg.EMOTION_LABELS[predicted.item()]
    prob_dict = {cfg.EMOTION_LABELS[i]: probabilities[0][i].item() for i in range(7)}

    print(f"\n[predict] 预测结果: {emotion} (置信度: {confidence.item():.4f})")
    print("[predict] 各类别概率:")
    for name, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name}: {prob:.4f} {bar}")

    return predicted.item()
