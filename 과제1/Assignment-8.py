import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 신경망 구조 설정
n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

# 신경망 구조 설계


def execute(initializer):
    mlp = Sequential()
    mlp.add(Dense(units=n_hidden1, activation='tanh', input_shape=(n_input,),
            kernel_initializer=initializer, bias_initializer='zeros'))
    mlp.add(Dense(units=n_hidden2, activation='tanh',
            kernel_initializer=initializer, bias_initializer='zeros'))
    mlp.add(Dense(units=n_hidden3, activation='tanh',
            kernel_initializer=initializer, bias_initializer='zeros'))
    mlp.add(Dense(units=n_hidden4, activation='tanh',
            kernel_initializer=initializer, bias_initializer='zeros'))
    mlp.add(Dense(units=n_output, activation='tanh',
            kernel_initializer=initializer, bias_initializer='zeros'))

    # 신경망 학습
    mlp.compile(loss='mean_squared_error', optimizer=Adam(
        learning_rate=0.001), metrics=['accuracy'])
    hist = mlp.fit(x_train, y_train, epochs=20, batch_size=128,
                   validation_data=(x_test, y_test), verbose=2)

    # 신경망의 정확률 측정
    res = mlp.evaluate(x_test, y_test, verbose=0)
    print("정확률은", res[1]*100, "%입니다.")

    # 정확률 곡선
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()

    # 손실 곡선
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid()
    plt.show()


execute('glorot_uniform')
execute('glorot_normal')
execute('random_normal')
