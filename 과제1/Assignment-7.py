from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
mnist = fetch_openml('mnist_784')
mnist.data = mnist.data/255.0
x_train = mnist.data[:60000]
x_test = mnist.data[60000:]
y_train = np.int16(mnist.target[:60000])
y_test = np.int16(mnist.target[60000:])

################
# 본인 코드 작성
################

# 은닉층이 1개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1

# 정확률
acc1 = 0
for i in range(10):
    acc1 += conf[i][i]
acc1 /= len(res)

# 은닉층이 2개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1

# 정확률
acc2 = 0
for i in range(10):
    acc2 += conf[i][i]
acc2 /= len(res)

# 은닉층이 3개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1

# 정확률
acc3 = 0
for i in range(10):
    acc3 += conf[i][i]
acc3 /= len(res)

# 은닉층이 4개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1

# 정확률
acc4 = 0
for i in range(10):
    acc4 += conf[i][i]
acc4 /= len(res)

# 은닉층이 5개인 경우
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), learning_rate_init=0.001,
                    batch_size=512, max_iter=20, solver='sgd', verbose=True)
mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

# 혼동 행렬
conf = np.zeros((10, 10), dtype=np.int16)
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1

# 정확률
acc5 = 0
for i in range(10):
    acc5 += conf[i][i]
acc5 /= len(res)


print("(은닉층 1개)테스트 집합에 대한 정확률은", acc1*100, "%입니다.")
print("(은닉층 2개)테스트 집합에 대한 정확률은", acc2*100, "%입니다.")
print("(은닉층 3개)테스트 집합에 대한 정확률은", acc3*100, "%입니다.")
print("(은닉층 4개)테스트 집합에 대한 정확률은", acc4*100, "%입니다.")
print("(은닉층 5개)테스트 집합에 대한 정확률은", acc5*100, "%입니다.")
