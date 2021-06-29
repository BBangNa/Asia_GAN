import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

# GAN 적대적 생성망
# auto encoder의 경우 학습한 이미지에서 가져옴 (창의력이 없다.)
# GAN의 경우 새로운 이미지를 만들어 낼 수 있다. (창의력이 있다.)
# G의 fake이미지를 D가 원본이미지와 구분할 수 없도록 만드는 것이 목표 (0과 1중에 1의 값이 나오도록 해야하는 것)

OUT_DIR = './OUT_img/'
img_shape = (28,28,1)  # 계속 쓸 거라서...
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100

(X_train , _), (_, _) = mnist.load_data()  # 검증 데이터도 필요없음
print(X_train.shape)  # (60000, 28, 28)이라는 결과값.

X_train = X_train / 127.5 - 1 # 마이너스 값이 나올 수도 있음 / 결과값이 -1~1사이의 값이 나올 수 있도록 한다.
X_train = np.expand_dims(X_train, axis=3)  # expand_dims : axis로 된 차원을 추가하는 함수 --> 그래서 ( , , ,1)이 생성됨!
print(X_train.shape)  # (60000, 28, 28, 1)이라는 결과값. reshape와 같은 결과

# build generator  +)두 개의 모델이 필요하다.
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise))  # 랜덤하게 만들어진 잡음 100개 --> 28*28*1=784가 되도록 함
generator_model.add(LeakyReLU(alpha=0.01))  # activate function이라서 layer하나만 넣는다. alpha값을 넣어야해서 activation을 따로 넣어준다.
generator_model.add(Dense(784, activation='tanh'))
generator_model.add(Reshape(img_shape))  # shape만 새로 잡아준 거라서 param 없음.

print(generator_model.summary())

# bulid discriminator
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))  # generator_model도 확인하기 위해서 한줄로 늘어뜨릴 필요가 있음.
                                                         # 원본 데이터는 한 줄로 늘어뜨려져 있기 때문.
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.01))  # LeakyRelu는 Relu랑 비슷하나 (-)값에 기울기를 줄 수 있기 때문에 사용함.
discriminator_model.add(Dense(1, activation='sigmoid'))
print(discriminator_model.summary()) # 출력은 하나가 되는 모델 생성.

discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['acc'])
discriminator_model.trainable = False # discriminator는 학습 시키지 않는다.

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())  # model 두 개 이어붙임.
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))  # batch_size개의 행, 1열
print(real)
fake = np.zeros((batch_size, 1)) # 해당 사이즈의 0으로만 가득찬 행렬을 만든다.
print(fake)

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]  # 128개의 이미지

    z = np.random.normal(0,1,(batch_size,noise)) # z는 noise 표준정규분포한것과 같다 평균0 표준편차1 주는것. (100개짜리 데이터 128개)
    fake_imgs = generator_model.predict(z) # 128장의 noise 이미지 완성

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real) # 따로 학습하기 때문에 trainable=False로 줘도 상관없음. real 이미지를 학습시켰을 때의 loss값
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake) # fake 이미지를 학습시켰을 때의 loss값

    d_loss , d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # 원본 이미지, 가짜 이미지의 loss값들의 평균값.
    discriminator_model.trainable = False

    z = np.random.normal(0, 1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real) # z라는 한계치를 뽑아 주고, batch에 대한 정답 라벨'real'을 줌
    # train_on_batch--> 한번만 학습하고 만다, 출력이 무조건 1이 나와야 한다.
    # 답이 1이 되도록 학습되어야 하기 때문에...

    if (itr)%sample_interval == 0:
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
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()







