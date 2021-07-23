import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape, Activation, BatchNormalization, Dropout, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100
# 0 ~ 9
num_class = 10

class DataSet:
    def __init__(self,num_label):
        self.num_label = num_label
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        def preprocess_img(x):
            x = (x.astype(np.float32) - 127.5) / 127.5 #0 ~ 255 --> -1 ~ 1
            x = np.expand_dims(x, axis=3) #28 X 28 --> 28 X 28 X 1
            return  x

        def preprocess_label(y):
            return y.reshape(-1,1)

        self.x_train = preprocess_img(self.x_train)
        self.y_train = preprocess_label(self.y_train)

        self.x_test = preprocess_img(self.x_test)
        self.y_test = preprocess_label(self.y_test)

    def batch_label(self, batch_size):
        idx = np .random.randint(0, self.num_label, batch_size)
        imgs = self.x_train[idx]
        labels =self.y_train[idx]
        return imgs, labels

    def batch_unlabel(self, batch_size):
        idx = np.random.randint(self.num_label, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_label)]
        y_train = self.y_train[range(self.num_label)]
        return x_train, y_train

    def testing_set(self):
        return self.x_test, self.y_test


def build_generator(z_dim):
    '''
    Generator
    z noise vector use FC and reshape to 7 X 7 X 256
    Transpose to 14 X 14 X 128
    BatchNormalization and LeakyReLU
    Transpose to 14 X 14 X 64 (same height and width: Conv2DTranspose stride = 1)
    BatchNormalization and LeakyReLU
    Transpose to 28 X 28 X 1 (same the image hight and width)
    tanh
    '''
    model = Sequential()
    # z noise vector use FC and reshape to 7 X 7 X 256
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # Transpose to 14 X 14 X 128
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))

    # BatchNormalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transpose to 14 X 14 X 64
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))

    # BatchNormalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transpose to 28 X 28 X 1
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding="same"))

    # Output layer with tanh activation
    model.add(Activation('tanh'))

    return model

def build_discriminator(img_shape):
    '''
        Discriminator
        input 28 X 28 X 1  output vector  true or false

        28 X 28 X 1  to 14 X 14 X 32
        LeakyReLU (not use BatchNormalization in this time)
        14 X 14 X 32 to 7 X 7 X 64
        BatchNormalization and LeakyReLU
        7 X 7 X 64 to 4 X 4 X 128
        BatchNormalization and LeakyReLU
        4 X 4 X 128 Flatten to 2048
        FC and sigmoid
    '''
    model = Sequential()

    # 28 X 28 X 1  to 14 X 14 X 32
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # 14 X 14 X 32 to 7 X 7 X 64
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))

    # BatchNormalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # 7 X 7 X 64 to 4 X 4 X 128
    model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))

    # BatchNormalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dropout(0.5))

    # Flatten
    model.add(Flatten())

    # Output layer with sigmoid activation
    model.add(Dense(num_class))

    return model


def build_discriminator_supervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    model.add(Activation('softmax'))

    return  model


def build_discriminator_unsupervised(discriminator_net):
    model = Sequential()

    model.add(discriminator_net)

    def predict(x):
        prediction =1.0 -(1.0/(backend.sum(backend.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction

    model.add(Lambda(predict))

    return model


def build_gan(generator, discriminator):

    model = Sequential()
    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model

# Build and compile the Discriminator
discriminator_net = build_discriminator(img_shape)

discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised .compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised .compile(loss='binary_crossentropy',optimizer=Adam())

discriminator_unsupervised.trainable = False

generator = build_generator(z_dim)

gan = build_gan(generator,discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

discriminator_all_supervised = build_discriminator_supervised(build_discriminator(img_shape))
discriminator_all_supervised.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

supervised_losses = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    real = np.ones((batch_size, 1))

    fake = np.zeros((batch_size, 1))
    for iteration in range(iterations):
        imgs, labels = dataset.batch_label(batch_size)
        labels = to_categorical(labels, num_classes=num_class)

        imgs_unlabel = dataset.batch_unlabel(batch_size)

        #生成假樣本
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        (d_loss_supervised, accuracy) = discriminator_supervised.train_on_batch(imgs, labels)
        (d_loss_supervised_all, accuracy_all) = discriminator_all_supervised.train_on_batch(imgs, labels)

        d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabel, real)
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)
        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train the Generator
        # ---------------------
        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))

        if (iteration + 1) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss supervised: %4f, acc.: %.2f%%] [D loss unsupervised: %4f] [G loss:%f]" %
                  (iteration + 1, d_loss_supervised, 100.0 * accuracy,d_loss_unsupervised, g_loss))


num_label =100
dataset = DataSet(num_label)


# Set hyperparameters
iterations = 8000
batch_size = 32
sample_interval = 800

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)

x, y = dataset.testing_set()
y = to_categorical(y, num_classes=num_class)

_, accuracy =discriminator_supervised.evaluate(x,y)
_, accuracy_all =discriminator_all_supervised.evaluate(x,y)

print("Test Accuracy: %.2f%%" % (100 * accuracy))
print("Test Accuracy (all supervised): %.2f%%" % (100 * accuracy_all))

