'''
06. MNIST 데이터셋에 대해 10-2.py를 10개 숫자 부류 중 
[2, 7] 두개 부류에만 적용하는 버전으로 수정하고 품질을 평가하기 위해 다음을 제시하시오. 
1) 마지막 epoch에서의 validation 데이터 셋의 mse loss 값 
2) x_test 원본 이미지, 생성된 이미지 10개씩 도시.
'''

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# MNIST 데이터를 읽고 신경망에 입력할 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

zdim = 32  # 잠복 공간의 차원

# 숫자 2,7에 대해서만 학습하기 위해 인덱스 필터링
f_train = np.logical_or(y_train == 2, y_train == 7)
f_test = np.logical_or(y_test == 2, y_test == 7)
x_train = x_train[f_train]
x_test = x_test[f_test]

# 오토인코더의 인코더 부분 설계
encoder_input = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu',
           padding='same', strides=(1, 1))(encoder_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Flatten()(x)
encoder_output = Dense(zdim)(x)
model_encoder = Model(encoder_input, encoder_output)
model_encoder.summary()

# 오토인코더의 디코더 부분 설계
decoder_input = Input(shape=(zdim,))
x = Dense(3136)(decoder_input)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu',
                    padding='same', strides=(1, 1))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu',
                    padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu',
                    padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(1, (3, 3), activation='relu',
                    padding='same', strides=(1, 1))(x)
decoder_output = x
model_decoder = Model(decoder_input, decoder_output)
model_decoder.summary()

# 인코더와 디코더를 결합하여 오토인코더 모델 구축
model_input = encoder_input
model_output = model_decoder(encoder_output)
model = Model(model_input, model_output)

# 오토인코더 학습
model.compile(optimizer='Adam', loss='mse')
model.fit(x_train, x_train, epochs=5, batch_size=128,
          shuffle=True, validation_data=(x_test, x_test))

# 복원 실험 1: x_test를 복원하는 예측 실험
decoded_img = model.predict(x_test)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_img[i].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
