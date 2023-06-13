'''
08. fashion MNIST에서 다른 패션 아이템 한 부류를 고른 다음 10-5.py을 실행하고 품질을 평가하기 위해 다음을 제시하시오. 
1) 마지막 iteration에서의 판별망(D)의 real data에 대한 분별 accuracy, fake data에 대한 분별 accuracy 
2) 학습을 마친 후 50개의 생성된 이미지 도시.
+8번 학습 시 iteration(교재 코드에서는 epoch로 표기됨) 값을 1000으로 낮춰 적용하세요.
'''
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, UpSampling2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train[np.isin(y_train, [1])]
x_train = (x_train.astype('float32')/255.0)*2.0-1.0  # [-1,1] 구간
x_test = (x_test.astype('float32')/255.0)*2.0-1.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

batch_siz = 64
epochs = 1000
dropout_rate = 0.4
batch_norm = 0.9
zdim = 100  # 잠복 공간의 차원

discriminator_input = Input(shape=(28, 28, 1))  # 분별망 D 설계
x = Conv2D(64, (5, 5), activation='relu', padding='same',
           strides=(2, 2))(discriminator_input)
x = Dropout(dropout_rate)(x)
x = Conv2D(64, (5, 5), activation='relu', padding='same', strides=(2, 2))(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(128, (5, 5), activation='relu', padding='same', strides=(2, 2))(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(128, (5, 5), activation='relu', padding='same', strides=(1, 1))(x)
x = Dropout(dropout_rate)(x)
x = Flatten()(x)
discriminator_output = Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_input, discriminator_output)

generator_input = Input(shape=(zdim,))  # 생성망 G 설계
x = Dense(3136)(generator_input)
x = BatchNormalization(momentum=batch_norm)(x)
x = Activation('relu')(x)
x = Reshape((7, 7, 64))(x)
x = UpSampling2D()(x)
x = Conv2D(128, (5, 5), padding='same')(x)
x = BatchNormalization(momentum=batch_norm)(x)
x = Activation('relu')(x)
x = UpSampling2D()(x)
x = Conv2D(64, (5, 5), padding='same')(x)
x = BatchNormalization(momentum=batch_norm)(x)
x = Activation('relu')(x)
x = Conv2D(64, (5, 5), padding='same')(x)
x = BatchNormalization(momentum=batch_norm)(x)
x = Activation('relu')(x)
x = Conv2D(1, (5, 5), activation='tanh', padding='same')(x)
generator_output = x
generator = Model(generator_input, generator_output)

discriminator.compile(
    optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
gan_input = Input(shape=(zdim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])


def train_discriminator(x_train):
    c = np.random.randint(0, x_train.shape[0], batch_siz)
    real = x_train[c]
    discriminator.train_on_batch(real, np.ones((batch_siz, 1)))

    p = np.random.normal(0, 1, (batch_siz, zdim))
    fake = generator.predict(p)
    discriminator.train_on_batch(fake, np.zeros((batch_siz, 1)))


def train_generator():
    p = np.random.normal(0, 1, (batch_siz, zdim))
    gan.train_on_batch(p, np.ones((batch_siz, 1)))

# 판별 정확도 계산


def calculate_discriminator_accuracy():
    discriminator.evaluate(x_test, np.ones((len(x_test), 1)))

    p = np.random.normal(0, 1, (len(x_test), zdim))
    fake = generator.predict(p)
    discriminator.evaluate(fake, np.zeros((len(x_test), 1)))

    real_acc = discriminator.predict(x_test)
    print("Real Data Accuracy:", np.sum(real_acc >= 0.5)/len(real_acc))

    p = np.random.normal(0, 1, (len(x_test), zdim))
    fake = generator.predict(p)
    fake_acc = discriminator.predict(fake)
    print('Fake Data Accuracy:', np.sum(fake_acc < 0.5)/len(fake_acc))


for i in range(1, epochs+1):  # 학습을 수행
    train_discriminator(x_train)
    train_generator()
    if (i % 100 == 0):  # 학습 도중 100세대마다 중간 상황 출력
        plt.figure(figsize=(20, 4))
        plt.suptitle('epoch '+str(i))
        for k in range(20):
            plt.subplot(2, 10, k+1)
            img = generator.predict(np.random.normal(0, 1, (1, zdim)))
            plt.imshow(img[0].reshape(28, 28), cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()

    if i == epochs:
        calculate_discriminator_accuracy()


imgs = generator.predict(np.random.normal(0, 1, (50, zdim)))
plt.figure(figsize=(20, 10))  # 학습을 마친 후 50개 샘플을 생성하여 출력
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(imgs[i].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
