import pandas as pd
import numpy as np

def load_data(csv_path="Back_end/data/train/fer2013.csv"):
    # 直接读取当前目录下的fer2013.csv
    print("正在加载本地FER2013数据集...")
    df = pd.read_csv(csv_path)

    # 处理图像数据
    pixels = df['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # 关键：直接获取类别标签，转成Long类型（不用one-hot）
    emotions = df['emotion'].values.astype(np.int64)

    # 划分训练/验证/测试集
    train_x, train_y = faces[df['Usage'] == 'Training'], emotions[df['Usage'] == 'Training']
    val_x, val_y = faces[df['Usage'] == 'PublicTest'], emotions[df['Usage'] == 'PublicTest']
    test_x, test_y = faces[df['Usage'] == 'PrivateTest'], emotions[df['Usage'] == 'PrivateTest']

    print(f"数据集加载完成！训练集：{len(train_x)}张，验证集：{len(val_x)}张，测试集：{len(test_x)}张")
    ans = (train_x, train_y, val_x, val_y, test_x, test_y)
    return ans
