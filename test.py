from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터셋 불러오기
mnist = fetch_openml('mnist_784')
mnist.data = mnist.data/255.0

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 나누기
x_train = mnist.data[:60000]
x_test = mnist.data[60000:]
y_train = np.int16(mnist.target[:60000])
y_test = np.int16(mnist.target[60000:])

# 은닉층이 1개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

# 정확률
acc = 0
for i in range(10):
    acc += conf[i][i]
acc /= len(res)
print(acc)
print("정확률은", acc*100, "%입니다.")

# 은닉층이 2개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print("은닉층이 2개인 경우 혼동 행렬:\n", conf)

# 정확률
acc = 0
for i in range(10):
    acc += conf[i][i]
acc /= len(res)
print("은닉층이 2개인 경우 정확률:", acc)
print("정확률은", acc*100, "%입니다.")

# 은닉층이 3개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print("\n은닉층이 3개인 경우 혼동 행렬:\n", conf)

# 정확률
acc = 0
for i in range(10):
    acc += conf[i][i]
acc /= len(res)
print("은닉층이 3개인 경우 정확률:", acc)
print("정확률은", acc*100, "%입니다.")

# 은닉층이 4개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print("\n은닉층이 4개인 경우 혼동 행렬:\n", conf)

# 정확률
acc = 0
for i in range(10):
    acc += conf[i][i]
acc /= len(res)
print("은닉층이 4개인 경우 정확률:", acc)
print("정확률은", acc*100, "%입니다.")

# 은닉층이 5개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print("\n은닉층이 5개인 경우 혼동 행렬:\n", conf)

# 정확률
acc = 0
for i in range(10):
    acc += conf[i][i]
acc /= len(res)
print("은닉층이 5개인 경우 정확률:", acc)
print("정확률은", acc*100, "%입니다.")
