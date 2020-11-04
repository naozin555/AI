import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras import models
import os

# macOS特有の警告文を非表示（GPUがないからCPUでやるときに出る）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _complement(hit):
    if len(str(hit)) < 2:
        return '000' + str(hit)
    elif len(str(hit)) < 3:
        return '00' + str(hit)
    elif len(str(hit)) < 4:
        return '0' + str(hit)
    else:
        return str(hit)


numbers4_df = pd.read_csv("NumbersResult/Numbers4_result.csv", )
numbers4_df["抽選日"] = numbers4_df["抽選日"].apply(lambda x: datetime.datetime.strptime(x, '%Y年%m月%d日'))
numbers4_df["当選数字"] = numbers4_df["当選数字"].apply(_complement)
numbers4_df.sort_index(ascending=False, inplace=True)

# 過去10回分の当選数字を考慮する
data_X = []
data_y = []
p = 10
for i in range(len(numbers4_df) - p):
    data_X.append(numbers4_df["当選数字"][i:i+p])
    data_y.append(numbers4_df["当選数字"][i+p])

data_X = np.array(data_X)
data_y = np.array(data_y)

# 訓練データとテストデータに分ける
len_train = int(len(data_X) * 0.7)

X_train = data_X[:len_train]
X_test = data_X[len_train:]
y_train = data_y[:len_train]
y_test = data_y[len_train:]

print(len(X_train), len(X_test))

# 正規化
scaler_x = MinMaxScaler()
X_train_n = scaler_x.fit_transform(X_train)
X_test_n = scaler_x.fit_transform(X_test)
scaler_y = MinMaxScaler()
y_train_n = scaler_y.fit_transform(y_train.reshape(len(y_train), 1))
y_test_n = scaler_y.fit_transform(y_test.reshape(len(y_test), 1))

print(X_train.shape, X_test.shape)

# データ形式の変換
# (データ数、系列数、説明変数の数)
X_train_n = np.reshape(X_train_n, (X_train_n.shape[0], 1, 10))
X_test_n = np.reshape(X_test_n, (X_test_n.shape[0], 1, 10))

# モデルの実装
n_hideen = 300
in_out_neurons = 1
model = models.Sequential()
model.add(
    LSTM(
        n_hideen,
        activation='tanh',
        input_shape=(1, p)
    )
)
model.add(Dense(1, activation='linear'))

# 学習設定
model.compile(
    loss='mean_squared_error',
    optimizer='sgd'
)

# モデルの学習
result = model.fit(
    X_train_n,
    y_train_n,
    batch_size=10,
    epochs=100
)

# 予測値の算出
y_prediction = model.predict(X_test_n)
# 正規化の復元
y_prediction = scaler_y.inverse_transform(y_prediction)
pred_result = []
for y_pred in y_prediction:
    for y in y_pred:
        pred_result.append(y)
        # 整数文字列にする場合
        # pred_result.append(str(int(y)))

print(f"Real: {y_test}")
print(f"Prediction: {pred_result}")
