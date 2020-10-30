import os


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from keras.layers import Input, Dense, Add, Multiply

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
image_size = 66

# データの読み込み
imagefiles = np.load("imagefiles_supplementary.npz")
X_train = imagefiles['X_train']
X_test = imagefiles['X_test']
y_train = imagefiles['y_train']
y_test = imagefiles['y_test']
# グレースケール画像をCNNに入力するための次元操作
X_train = X_train.reshape((-1, image_size, image_size, 1))
X_test = X_test.reshape((-1, image_size, image_size, 1))
# データの正規化
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# OneHotVector化する(正解ラベルの位置に1がつく)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


def _build(_input, *nodes):
    x = _input
    for node in nodes:
        if callable(node):
            x = node(x)
        elif isinstance(node, list):
            x = [_build(x, branch) for branch in node]
        elif isinstance(node, tuple):
            x = _build(x, *node)
        else:
            x = node
    return x


_input = Input(X_train.shape[1:])
output = _build(
    _input,
    # Reduction dual-path module×３の定義
    # ---------------------------
    # 畳み込み層の追加(96:フィルタ数)
    # バッチ正規化
    # 活性化関数：ReLu
    # ---------------------------
    # MaxPooling
    # ---------------------------
    # Reduction dual-path module1
    [(Conv2D(96, (3, 3), strides=(2, 2)),
      BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                         beta_initializer='zeros', gamma_initializer='ones',
                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                         beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                         gamma_constraint=None),
      Activation('relu')),
     MaxPooling2D(pool_size=(3, 3), strides=(2, 2))],
    # Reduction dual-path module2
    Add(),
    [(Conv2D(96, (3, 3), strides=(2, 2)),
      BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                         beta_initializer='zeros', gamma_initializer='ones',
                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                         beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                         gamma_constraint=None),
      Activation('relu')),
     MaxPooling2D(pool_size=(3, 3), strides=(2, 2))],
    # Reduction dual-path module3
    Add(),
    [(Conv2D(96, (3, 3), strides=(2, 2)),
      BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                         beta_initializer='zeros', gamma_initializer='ones',
                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                         beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                         gamma_constraint=None),
      Activation('relu')),
     MaxPooling2D(pool_size=(3, 3), strides=(2, 2))],

    # Dual-path modules×10の定義
    # ---------------------------
    # 畳み込み層の追加(112:フィルタ数)
    # バッチ正規化
    # 活性化関数：ReLu
    # ---------------------------
    # Dual-path modules2の定義
    # 畳み込み層の追加(48:フィルタ数)
    # バッチ正規化
    # 活性化関数：ReLu
    # ---------------------------
    # Dual-path modules1
    Add(),
    [(Conv2D(112, (1, 1), strides=(1, 1)),
      BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                         beta_initializer='zeros', gamma_initializer='ones',
                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                         beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                         gamma_constraint=None),
      Activation('relu'),
      ),
     (Conv2D(48, (3, 3), strides=(1, 1)),
      BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                         beta_initializer='zeros', gamma_initializer='ones',
                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                         beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                         gamma_constraint=None),
      Activation('relu'),
      )],
    # # Dual-path modules2
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules3
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules4
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules5
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules6
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules7
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules8
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules9
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # # Dual-path modules10
    # Add(),
    # [(Conv2D(112, (1, 1), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu')),
    #  (Conv2D(48, (3, 3), strides=(1, 1)),
    #   BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    #                      beta_initializer='zeros', gamma_initializer='ones',
    #                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
    #                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    #                      gamma_constraint=None),
    #   Activation('relu'))],
    # 全結合
    Add(),
    [MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
     Flatten(),
     Dense(256, activation='relu'),
     Dropout(0.5),
     Dense(num_classes, activation='softmax')
     ]
)
model = Model(_input, output)
model.summary()

# # 損失関数の設定
# opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# # トレーニングの実施
# # 学習
# print("start training")
# hist = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))
# # 評価
# print("start eval")
# score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)  # verbose:途中結果表示
# print('Test Loss: ', score[0])
# print('Test Accuracy: ', score[1])
#
# model.save('leukemia_cnn_supplementary.h5')
#
# # 学習の様子をグラフへ描画
# # 正解率の推移をプロット
# fig = plt.figure()
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Accuracy')
# plt.legend(['train', 'test'], loc='upper left')
# fig.savefig('result/cnn_supplementary/cnn_accuracy_supplementary.png')
# plt.close()
# # ロスの推移をプロット
# fig = plt.figure()
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Loss')
# plt.legend(['train', 'test'], loc='upper left')
# fig.savefig('result/cnn_supplementary/cnn_loss_supplementary.png')
# plt.close()
# # Confusion matrix作成
# plt.figure()
# y_pred = model.predict(X_test)
# y_test = imagefiles['y_test']  # one hot vector化されているのでロードし直す
# cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
# ticklabels = ["blasts_highSSC_granulocytes",
#               "blasts_highSSC_middle_ugly",
#               "blasts",
#               "normal cells",
#               "blasts_highSSC_upper_dead"]
# sns.heatmap(cm, annot=True, cmap='Blues', yticklabels=ticklabels, xticklabels=ticklabels)
# plt.ylabel("Correct")
# plt.xlabel("Prediction")
# plt.tight_layout()
# plt.savefig('result/cnn_supplementary/confusion_matrix_cnn_supplementary.png')
# plt.close()
#
# # F1 micro/macro
# f1_macro = f1_score(y_test, np.argmax(y_pred, axis=1), average="macro")
# f1_micro = f1_score(y_test, np.argmax(y_pred, axis=1), average="micro")
# print(f"f1_macro:{f1_macro}")
# print(f"f1_miro:{f1_micro}")
