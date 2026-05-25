"""
预测模块 - 单张图片情感预测
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import IMG_SIZE, NUM_CLASSES, DEVICE, MODEL_SAVE_PATH, EMOTION_LABELS
from model import EmotionCNN


def predict_image(image_path, model_path=None):
    """
    对单张图片进行情感预测
    参数:
        image_path: 图片路径
        model_path: 模型权重路径 (默认使用最优模型)
    返回:
        预测的情感类别名称和各类别概率
    """
    if model_path is None:
        model_path = MODEL_SAVE_PATH

    if not os.path.exists(image_path):
        print(f"[predict] 错误: 图片不存在 - {image_path}")
        return None, None

    if not os.path.exists(model_path):
        print(f"[predict] 错误: 模型文件不存在 - {model_path}")
        print("[predict] 请先训练模型")
        return None, None

    # 加载模型
    model = EmotionCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # 预处理图片
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    emotion = EMOTION_LABELS[predicted.item()]
    prob_dict = {EMOTION_LABELS[i]: probabilities[0][i].item() for i in range(NUM_CLASSES)}

    print(f"\n[predict] 预测结果: {emotion} (置信度: {confidence.item():.4f})")
    print("[predict] 各类别概率:")
    for name, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name}: {prob:.4f} {bar}")

    return emotion, prob_dict
