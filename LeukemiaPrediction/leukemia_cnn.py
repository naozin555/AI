import os


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

# macOS特有の警告文を非表示（GPUがないからCPUでやるときに出る）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# パラメータの初期化
classes = [
    "normal cells",
    "blasts",
    "blasts_highSSC_granulocytes",
    "blasts_highSSC_middle_ugly",
    "blasts_highSSC_upper_dead",
]
num_classes = len(classes)
image_size = 40

# データの読み込み
imagefiles = np.load("imagefiles.npz")
X_train = imagefiles['X_train']
X_test = imagefiles['X_test']
y_train = imagefiles['y_train']
y_test = imagefiles['y_test']
# グレースケール画像をCNNに入力するための次元操作
X_train = X_train.reshape((-1, 40, 40, 1))
X_test = X_test.reshape((-1, 40, 40, 1))
# データの正規化
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# OneHotVector化する(正解ラベルの位置に1がつく)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# モデルの定義
# Sequencialモデル(NNのモデルを造るという)の宣言
model = Sequential()
# 畳み込み層の追加(64:フィルタ数),活性化関数はRelu，入力データ形式
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
# 2層目の畳み込み層(入力層の形式input_shapeは必要ない)
model.add(Conv2D(32, (3, 3), activation='relu'))
# データの圧縮
model.add(MaxPooling2D(pool_size=(2, 2)))
# ドロップアウト
model.add(Dropout(0.25))
# 精度向上のため畳み込み層の追加
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(72, (3, 3), activation='relu'))
model.add(Conv2D(72, (3, 3), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.25))

# 全結合の実施
# データを直列に並べる
model.add(Flatten())
# 直列データから全結合への中間層
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# 出力層
model.add(Dense(num_classes, activation='softmax'))

# 損失関数の設定
opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# トレーニングの実施
# 学習
print("start training")
hist = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))
# 評価
print("start eval")
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)  # verbose:途中結果表示
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

model.save('leukemia_cnn.h5')

# 学習の様子をグラフへ描画
# 正解率の推移をプロット
fig = plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('result/cnn/cnn_accuracy.png')
plt.close()
# ロスの推移をプロット
fig = plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('result/cnn/cnn_loss.png')
plt.close()
# Confusion matrix作成
plt.figure()
y_pred = model.predict(X_test)
y_test = imagefiles['y_test']  # one hot vector化されているのでロードし直す
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
ticklabels = ["blasts_highSSC_granulocytes",
              "blasts_highSSC_middle_ugly",
              "blasts",
              "normal cells",
              "blasts_highSSC_upper_dead"]
sns.heatmap(cm, annot=True, cmap='Blues', yticklabels=ticklabels, xticklabels=ticklabels)
plt.ylabel("Correct")
plt.xlabel("Prediction")
plt.tight_layout()
plt.savefig('result/cnn/confusion_matrix_cnn.png')
plt.close()

# F1 micro/macro
f1_macro = f1_score(y_test, np.argmax(y_pred, axis=1), average="macro")
f1_micro = f1_score(y_test, np.argmax(y_pred, axis=1), average="micro")
print(f"f1_macro:{f1_macro}")
print(f"f1_miro:{f1_micro}")
