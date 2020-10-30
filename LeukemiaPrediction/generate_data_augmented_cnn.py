import glob

from PIL import Image
import numpy as np
from sklearn import model_selection

# パラメータの初期化
classes = [
    "blasts_highSSC_granulocytes",
    "blasts_highSSC_middle_ugly",
    "blasts",
    "normal cells",
    "blasts_highSSC_upper_dead",
]
num_classes = len(classes)
image_size = 40
num_testdata = 200

# 画像ファイルを格納するためのリスト
X_train = []
X_test = []
# 正解ラベル
Y_train = []
Y_test = []

data_array = []
# 画像の読み込みとNumPy配列への変換
for index, classlabel in enumerate(classes):

    if index == 0:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch1_*.tif")
        f2 = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch6_*.tif")
        files.extend(f2)
    elif index == 1:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1/ch1_*.tif")
        f2 = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1/ch6_*.tif")
        files.extend(f2)
    elif index == 2:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch1_*.tif")
        f2 = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch6_*.tif")
        files.extend(f2)
    elif index == 3:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_focused _ R6 _ R7 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch1_*.tif")
        f2 = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_focused _ R6 _ R7 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch6_*.tif")
        files.extend(f2)
    elif index == 4:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch1_*.tif")
        f2 = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/ch6_*.tif")
        files.extend(f2)

    for i, file in enumerate(files):

        # テスト用にデータを確保(200画像/クラス)
        if i < num_testdata:
            image = Image.open(file)        # 画像ファイルを開く
            image = image.resize((image_size, image_size))  # (念のため)サイズ調整
            data = np.asarray(image)
            X_test.append(data)
            Y_test.append(index)

        # 学習用にデータを確保
        elif index == 0:
            image = Image.open(file)
            image = image.resize((image_size, image_size))
            # 学習データの水増し(4倍)
            # 回転
            for angle in range(-360, 0, 90):
                img_rotated = image.rotate(angle)
                rotated_data = np.asarray(img_rotated)
                X_train.append(rotated_data)
                Y_train.append(index)
        elif index == 1:
            # 学習データの水増し(34倍)
            image = Image.open(file)
            image = image.resize((image_size, image_size))
            for angle in range(-80, 81, 10):
                # 回転(17倍)
                img_rotated = image.rotate(angle)
                rotated_data = np.asarray(img_rotated)
                X_train.append(rotated_data)
                Y_train.append(index)
                # 反転(2倍)
                img_trans = image.transpose(Image.FLIP_TOP_BOTTOM)
                rotated_data = np.asarray(img_trans)
                X_train.append(rotated_data)
                Y_train.append(index)
        elif index == 2:
            image = Image.open(file)
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X_train.append(data)
            Y_train.append(index)
        elif index == 3:
            image = Image.open(file)
            image = image.resize((image_size, image_size))
            # 学習データの水増し(94倍)
            for angle in range(-92, 93, 4):
                # 回転(47倍)
                img_rotated = image.rotate(angle)
                rotated_data = np.asarray(img_rotated)
                X_train.append(rotated_data)
                Y_train.append(index)
                # 反転(2倍)
                img_trans = image.transpose(Image.FLIP_TOP_BOTTOM)
                rotated_data = np.asarray(img_trans)
                X_train.append(rotated_data)
                Y_train.append(index)

        elif index == 4:
            image = Image.open(file)
            image = image.resize((image_size, image_size))
            # 学習データの水増し(78倍)
            for angle in range(-76, 77, 4):
                # 回転(39倍)
                img_rotated = image.rotate(angle)
                rotated_data = np.asarray(img_rotated)
                X_train.append(rotated_data)
                Y_train.append(index)
                # 反転(2倍)
                img_trans = image.transpose(Image.FLIP_TOP_BOTTOM)
                rotated_data = np.asarray(img_trans)
                X_train.append(rotated_data)
                Y_train.append(index)
    print(f"{index}: {len(X_train)}")

# NumPy配列にする
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

for xy_train in [X_train, y_train]:
    np.random.seed(1)
    np.random.shuffle(xy_train)


# データを学習用とテスト用に分割して，データを保存する
np.savez("imagefiles_augmented.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
