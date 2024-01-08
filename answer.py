# p18 の問題に対する解答例です。

# numpy を np という名前でインポートする
import numpy as np

# 0から7まで1ずつ増える等差数列を作成する
a = np.arange(8)
print(a)

# 5から9まで1ずつ増える等差数列を作成する
b = np.arange(5, 10)
print(b)

# 0 から 1 まで 0.1 ずつ増える等差数列を作成する
c = np.arange(0, 1.1, 0.1)
print(c)

# p32 の問題に対する解答例です。

# 3x3 の単位行列を作成する
d = np.eye(3)
print(d)

# a を10倍したものを作成する
e = d * 10
print(e)

# a に e を掛ける
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
f = a * e
print(f)

# f の右上の2x2 の部分行列を取り出す
g = f[:2, 1:]
print(g)

# g を 4x1 の行列に変換する
h = g.reshape(4, 1)
print(h)

# p43 の問題に対する解答例です。

# 1-1 numpy を np という名前でインポートする
import numpy as np

# 1-2 4x4 の単位行列を作成する
a = np.eye(4)
print(a)

# 1-3 行列bを作成する
b = np.arange(1, 17).reshape(4, 4)
print(b)

# 1-4 b の0行目をc，1列目をdとする
c = b[0, :]
d = b[:, 1]
print(c)
print(d)

# 1-5 1 ではなく 1:2 を使って列ベクトルを作成する
e = b[:, 1:2]
print(e)

# 1-6 a とbを足し合わせてfを作成する
f = a + b
print(f)

# 1-7 f の3行2列目の要素を取り出す
print(f[3, 2])

# 1-8 f の各要素についてcosを計算しgを作成する
g = np.cos(f)
print(g)

# 1-9 gを横方向に和を取ったベクトルhを作成する
h = np.sum(g, axis=1)
print(h)

# 1-10 g を縦方向に和を取ったベクトルiを作成する
i = np.sum(g, axis=0)
print(i)

# 1-11 hのうち0以上の要素だけを持つベクトルjを作成する
# （問題文ではiとなっていますが正しくはhです．iのときはjは空になります）
j = h[h >= 0]
print(j)

# 1-12 iのうち0以上の要素の和を求める
k = np.sum(j)
print(k)

# 2-1 numpy と matplotlib.pyplot をインポートする
import numpy as np
import matplotlib.pyplot as plt

# 2-2 -5から5まで0.01刻みの等差数列を作成する
x = np.arange(-5, 5, 0.01)
print(x)

# 2-3 x の各要素についてsin2xを計算しyを作成する
y = np.sin(2 * x)
print(y)

# 2-4 x と y をプロットする
plt.plot(x, y)
plt.show()

# 2-5 2-4 のx軸方向の表示範囲を -π から π にする
plt.xlim(-np.pi, np.pi)
plt.plot(x, y)
plt.show()

# 2-6 2-5 のグラフにタイトルをつける
plt.title("sin(2x)")
plt.plot(x, y)
plt.show()

# 課題3
# 3-1 -5から5まで0.01刻みの等差数列を作成する
x = np.arange(-5, 5, 0.01)

# 3-2 y=e^x sin(4x) を計算する
y = np.exp(x) * np.sin(4 * x)

# 3-3 x軸は -2π から 2π までの範囲でグラフを描く
plt.xlim(-2 * np.pi, 2 * np.pi)

# 3-4 x軸とy軸に x と y のラベルをつける
plt.xlabel("x")
plt.ylabel("y")

# 3-5 グラフに罫線をつける
plt.grid()

# 3-6 グラフを表示
plt.plot(x, y)
plt.show()

# p88-92の問題に対する解答例です。

# 1-1 CSVファイルを読み込む
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/yu-suzuki/ds2023/main/titanic.csv")
print(df)

# 1-2 df を DataFrameからndarrayに変換する
data = df.values

# 1-3 正解ラベル（1列目）をyに代入する
y = data[:, 0]
print(y)

# 1-4 正解ラベルをint型に変更する
y = data[:, 0].astype(dtype=int)
print(y)

# 1-5 特徴量（2〜7列目）を変数Xに入れる
X = data[:, 2:8]
print(X)

# 1-6 Xとyを訓練データとテストデータに分割する
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 1-7 .shapeを使ってX_train, X_test, y_train, y_testの形状を確認する
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1-8 X_trainとy_trainの0行目を表示する
print(X_train[0])
print(y_train[0])

# 2-1 KNeighborsClassifierをインポートする
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=1)

# 2-2 訓練データを使ってnnを訓練する
nn.fit(X_train, y_train)

# 2-3 テストデータのラベルを予測する
y_pred = nn.predict(X_test)

# 2-4 y_predとy_testを比較して正解率を計算する
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# 3-1 KNeighborsClassifierをインポートする
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=3)

# 3-2 訓練データを使ってnnを訓練する
nn.fit(X_train, y_train)

# 3-3 テストデータのラベルを予測する
y_pred = nn.predict(X_test)

# 3-4 y_predとy_testを比較して正解率を計算する
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# 4-1 MLPClassifierをインポートする
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu', batch_size=255, max_iter=300)

# 4-2 訓練データを使ってmlpを訓練する
mlp.fit(X_train, y_train)

# 4-3 テストデータのラベルを予測する
y_pred = mlp.predict(X_test)

# 4-4 y_predとy_testを比較して正解率を計算する
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)