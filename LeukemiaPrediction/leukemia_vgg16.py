import os


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.applications import VGG16
import matplotlib.pyplot as plt


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
image_size = 224

# データの読み込み
imagefiles = np.load("imagefiles_vgg16.npz")
X_train = imagefiles['X_train']
X_test = imagefiles['X_test']
y_train = imagefiles['y_train']
y_test = imagefiles['y_test']
# グレースケール画像をCNNに入力するための次元操作
X_train = X_train.reshape((-1, 224, 224, 1))
X_test = X_test.reshape((-1, 224, 224, 1))
# データの正規化
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# OneHotVector化する(正解ラベルの位置に1がつく)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# モデルの定義
# VGG16のNN
model = VGG16(weights=None, include_top=False, input_shape=X_train.shape[1:])

# トップモデルの定義
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

model = Model(inputs=model.input, outputs=top_model(model.output))

# ハイパーパラメータの最適化
for layer in model.layers[:15]:
    layer.trainable = False
# 学習率を0.01でスタート
# opt = SGD(lr=0.01)
opt = Adam()
# 損失関数の設定
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# トレーニングの実施
# 学習
hist = model.fit(X_train, y_train, batch_size=32, epochs=30)
# 評価
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)  # verbose:途中結果表示
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

# 学習の様子をグラフへ描画
# 正解率の推移をプロット
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ロスの推移をプロット
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('leukemia_vgg16.h5')
