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
image_size = 66

# 画像の読み込みとNumPy配列への変換
X = []  # 画像ファイルを格納するためのリスト
Y = []  # 正解ラベル(carなのかmotorbikeなのか＝0なのか1なのか)

for index, classlabel in enumerate(classes):

    if index == 0:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/*.tif")
    elif index == 1:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1/*.tif")
    elif index == 2:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/*.tif")
    elif index == 3:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_focused _ R6 _ R7 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/*.tif")
    elif index == 4:
        files = glob.glob(
            f"leukemia/LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ {classlabel} _ cd 19 pos _ R2 _ R1_minh/*.tif")

    # 各ファイルに下記の処理を施す
    for i, file in enumerate(files):
        if "ch1_" in file or "ch6_" in file:
            image = Image.open(file)        # 画像ファイルを開く
            image = image.resize((image_size, image_size))  # (念のため)サイズ調整
            data = np.asarray(image)
            X.append(data)
            Y.append(index)

# NumPy配列にする
X = np.array(X)
Y = np.array(Y)
np.set_printoptions(threshold=np.inf)
for xy in [X, Y]:
    np.random.seed(1)
    np.random.shuffle(xy)

# データを学習用とテスト用に分割して，データを保存する
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
np.savez("imagefiles_supplementary.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
