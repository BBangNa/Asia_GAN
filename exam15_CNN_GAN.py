import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):  # 해당 경로에 폴더가 없으면 만들라는 명령
    os.makedirs(OUT_DIR)
img_shape = (28,28,1)
epoch = 50000
batch_size = 128
noise = 100
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data()
#print(X_train.shape)

X_train = X_train / 127.5 - 1
X_train = np.expand_dims(X_train, axis=3)
#print(X_train.shape)

# build generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise)) # 사이즈 뻥튀기하기
generator_model.add(Reshape((7,7,256)))  # shape을 하나의 튜플로 묶어서 줘야함. / (7*7)이미지 256개

generator_model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')) # ..?
generator_model.add(BatchNormalization()) # ...?
generator_model.add(LeakyReLU(alpha=0.01)) # (-)값을 활용하기 위해서 사용함.

generator_model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same')) # strides=2는 이미지 픽셀을 두배로 해준다는 뜻~!
generator_model.add(Activation('tanh'))

generator_model.summary()

# build discriminator

discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
discriminator_model.summary()

"""
이미지가 복잡할 수록 레이어가 딥해져야 한다. 모델을 깊고 크게 쌓아야한다.
( 레이어가 적으면 정확도가 떨어지고, 레이어가 많으면 시간이 오래 걸린다. 하지만 정확도는 올라간다. )
"""

discriminator_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
discriminator_model.trainable = False

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))
print(real)
fake = np.zeros((batch_size, 1))
print(fake)

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    z = np.random.normal(0,1,(batch_size,noise)) # z는 noise 표준정규분포한것과 같다 평균0 표준편차1 주는것. (100개짜리 데이터 128개)
    fake_imgs = generator_model.predict(z) # 128장의 noise 이미지 완성

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real) # 따로 학습하기 때문에 trainable=False로 줘도 상관없음. real 이미지를 학습시켰을 때의 loss값
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake) # CNN으로 학습시 학습이 너무 잘 되기 때문...

    d_loss , d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # 원본 이미지, 가짜 이미지의 loss값들의 평균값.
    #discriminator_model.trainable = False # 한번만 넣으면 되기 때문에 빼도 괜찮음^^;;


    z = np.random.normal(0,1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real) # z라는 한계치를 뽑아 주고, batch에 대한 정답 라벨'real'을 줌
    # train_on_batch--> 한번만 학습하고 만다, 출력이 무조건 1이 나와야 한다.
    # 답이 1이 되도록 학습되어야 하기 때문에...

    if itr%sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(itr, d_loss, d_acc*100, gan_hist)) # 소숫점 아래 2자리까지 볼 것.
        row = col = 4
        z = np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off') # x,y축 눈금 없애기
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+2))
        plt.savefig(path)
        plt.close()

# accuracy가 높다고 다 좋은게 아니다. Discriminator_model의 정확률이 높으면















