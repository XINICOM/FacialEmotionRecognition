import torch
import numpy as np
from PIL import Image
import cv2
import os

EMOTIONS = ["愤怒", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性"]

def get_face_detector():
    # OpenCV 自带的 Haar 特征文件路径
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise ValueError(f"Cascade file {cascade_path} does not exist")
    return cv2.CascadeClassifier(cascade_path)

face_cascade = get_face_detector()


def predict_CNN(model, img_path, device):
    """
    使用训练好的 CNN 模型预测单张图片中人脸的表情
    :param model: 训练好的 EmotionCNN 模型
    :param img_path: 图片路径(str)
    :param device: 运行设备 (torch.device 或 'cpu'/'cuda')
    :return: 预测的表情标签字符串, 裁剪缩放后的人脸矩阵
    """
    # 1. 确保模型在正确的设备上,并切换到评估模式
    model = model.to(device)
    model.eval()  # 关闭 Dropout 等行为

    # 2. 读取图片,并转为 OpenCV 灰度图（Haar 检测需要灰度）
    img = Image.open(img_path)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 3. 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        raise ValueError("未在输入图片中检测到人脸")

    # 4. 取面积最大的人脸并进行前处理
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_gray = gray[y:y + h, x:x + w]

    # 缩放到 48x48
    face_resized = cv2.resize(face_gray, (48, 48))
    # 转为 Tensor 并归一化 [1, 1, 48, 48]
    input_tensor = torch.tensor(face_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    # 5. 禁用梯度进行模型推理
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()

    # 打印并返回结果
    emotion_result = EMOTIONS[pred_idx]
    print(f"预测表情：{emotion_result}")
    return emotion_result, face_resized