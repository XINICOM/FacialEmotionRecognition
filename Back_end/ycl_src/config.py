"""
全局配置文件 - 人脸情感识别项目
"""
import os

# 解决 Windows 上 OpenMP 重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fer2013.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../checkpoints")
CONTROL_DIR = os.path.join(BASE_DIR, "../.control")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ==================== 数据配置 ====================
IMG_SIZE = 48
NUM_CLASSES = 7
EMOTION_LABELS = {
    0: "生气", 1: "厌恶", 2: "害怕",
    3: "快乐", 4: "悲伤", 5: "惊讶", 6: "中性"
}

# ==================== 训练配置 ====================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
VAL_SPLIT_RATIO = 0.1

# ==================== 设备配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 控制标志文件 ====================
PAUSE_FLAG = os.path.join(CONTROL_DIR, "pause.flag")
RESUME_FLAG = os.path.join(CONTROL_DIR, "resume.flag")
TERMINATE_FLAG = os.path.join(CONTROL_DIR, "terminate.flag")

# 确保目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CONTROL_DIR, exist_ok=True)
