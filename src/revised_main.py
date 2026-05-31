import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
# =====================================================
# 配置
# =====================================================
import src.config as cfg

DATA_PATH = cfg.DATA_PATH
IMAGE_SIZE = cfg.IMAGE_SIZE
BATCH_SIZE = cfg.BATCH_SIZE
EPOCHS = cfg.EPOCHS
LEARNING_RATE = cfg.LEARNING_RATE
NUM_CLASSES = cfg.NUM_CLASSES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = cfg.MODEL_SAVE_PATH
EMOTIONS = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]
# =====================================================
# 数据预处理
# =====================================================
def preprocessing(image):
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5]*3,
            [0.5]*3
        )
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    return train_transform, test_transform
# =====================================================
# 数据集加载
# =====================================================
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, transform=None, usage='Training'):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emotion = int(row['emotion'])
        pixels = np.array(row['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, emotion
# =====================================================
# Transformer Patch Embedding
# =====================================================
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=6,
                 emb_size=512,
                 img_size=48):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_size)
        )
        self.positions = nn.Parameter(
            torch.randn(num_patches + 1, emb_size)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
# =====================================================
# Emotion Transformer
# =====================================================
class EmotionTransformer(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.embedding = PatchEmbedding()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        return output
# =====================================================
# Pause / Resume / Terminate
# =====================================================
PAUSE_FILE = 'pause.flag'
TERMINATE_FILE = 'terminate.flag'
def check_pause():
    while os.path.exists(PAUSE_FILE):
        print("训练暂停中...")
        time.sleep(2)
def check_terminate():
    if os.path.exists(TERMINATE_FILE):
        print("训练终止")
        return True
    return False
# =====================================================
# 训练函数
# =====================================================
def train_model():
    train_dataset = FER2013Dataset(
        DATA_PATH,
        transform=train_transform,
        usage='Training'
    )
    val_dataset = FER2013Dataset(
        DATA_PATH,
        transform=test_transform,
        usage='PublicTest'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    model = EmotionTransformer(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
    best_acc = 0
    for epoch in range(EPOCHS):
        if check_terminate():
            break
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            check_pause()
            if check_terminate():
                return
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Loss: {total_loss:.4f}")
        # =========================
        # 验证
        # =========================
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                truths.extend(labels.numpy())
        acc = accuracy_score(truths, preds)
        print(f"Validation Accuracy: {acc:.4f}")
        # 保存当前模型
        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                'best_model.pth'
            )
            print("保存最佳模型")
    print(f"\n训练完成")
    print(f"最佳验证准确率: {best_acc:.4f}")
# =====================================================
# 图片预测
# =====================================================
def predict_emotion(image_path):
    model = EmotionTransformer(NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            'best_model.pth',
            map_location=DEVICE
        )
    )
    model.to(DEVICE)
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    emotion = EMOTIONS[predicted.item()]
    print(f"\n预测情感: {emotion}")
# =====================================================
# 主程序
# =====================================================
while True:
    print("\n========== 人脸情感识别系统 ==========")
    print("1. 开始训练")
    print("2. 暂停训练")
    print("3. 恢复训练")
    print("4. 终止训练")
    print("5. 图片预测")
    print("6. 退出")
    choice = input("\n请选择功能: ")
    # ==================================
    # 开始训练
    # ==================================
    if choice == '1':
        # 删除旧终止标记
        if os.path.exists(TERMINATE_FILE):
            os.remove(TERMINATE_FILE)
        train_model()
    # ==================================
    # 暂停训练
    # ==================================
    elif choice == '2':
        open(PAUSE_FILE, 'w').close()
        print("训练已暂停")
    # ==================================
    # 恢复训练
    # ==================================
    elif choice == '3':
        if os.path.exists(PAUSE_FILE):
            os.remove(PAUSE_FILE)
        print("训练继续")
    # ==================================
    # 终止训练
    # ==================================
    elif choice == '4':
        open(TERMINATE_FILE, 'w').close()
        print("训练终止")
    # ==================================
    # 图片预测
    # ==================================
    elif choice == '5':
        image_path = input("请输入图片路径: ")
        predict_emotion(image_path)
    # ==================================
    # 退出
    # ==================================
    elif choice == '6':
        print("程序退出")
        break
    else:
        print("输入无效，请重新选择")