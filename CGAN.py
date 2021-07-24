import matplotlib.pyplot as plt
import numpy as np

from keras import backend
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape, Activation, BatchNormalization, Dropout, Input, Lambda, Embedding,Multiply,Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential,Model
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

def build_generator(z_dim):
    '''
    要將 label 和 noise 作 element wise
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

def build_cgan_generator(z_dim):
    # combine noise and label
    # then put in original gan
    z = Input(shape=(z_dim,))

    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_class, z_dim, input_length=1)(label)

    label_embedding = Flatten()(label_embedding)

    joined_representation = Multiply()([z, label_embedding])

    generator = build_generator(z_dim)

    conditioned_img = generator(joined_representation)

    return Model([z, label], conditioned_img)


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

    # 28 X 28 X 2  to 14 X 14 X 64
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(img_shape[0], img_shape[1], img_shape[2] + 1), padding="same"))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # 14 X 14 X 64 to 7 X 7 X 64
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

    # Flatten
    model.add(Flatten())

    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    return model

def build_cgan_discriminator(img_shape):

    img = Input(shape=img_shape)

    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_class, np.prod(img_shape), input_length=1)(label)#np.prod(img_shape)= 28 X 28 X 2 = 784

    label_embedding = Flatten()(label_embedding)

    label_embedding = Reshape(img_shape)(label_embedding)

    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = build_discriminator(img_shape)

    classification = discriminator(concatenated)

    return Model([img, label], classification)

def build_cgan(generator, discriminator):
     z = Input(shape=(z_dim, ))

     labels = Input(shape=(1, ))

     img = generator([z, labels])

     classification = discriminator([img, labels])

     model = Model([z, labels], classification)

     return  model


discriminator = build_cgan_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

discriminator.trainable = False
generator = build_cgan_generator(z_dim)
cgan = build_cgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []

def train(iterations, batch_size, sample_interval):
    # Load the MNIST dataset
    (X_train, Y_train), (_, _) = mnist.load_data()
    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
    # Labels for real images: all ones
    real = np.ones((batch_size, 1))
    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))
    for iteration in range(iterations):
        # -------------------------
        #  Train the Discriminator
        # -------------------------
        # Get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, label= X_train[idx], Y_train[idx]
        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict([z,label])
        # Train Discriminator
        d_loss_real = discriminator.train_on_batch([imgs,label], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs,label], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train the Generator
        # ---------------------
        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        labels =np.random.randint(0,num_class, batch_size).reshape(-1, 1)
        # Train Generator
        g_loss = cgan.train_on_batch([z,labels], real)
        if (iteration + 1) % sample_interval == 0:

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss[0], 100.0 * d_loss[1], g_loss))

            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])
            # Output a sample of generated image
            sample_images()

def sample_images(image_grid_rows=2, image_grid_columns=5):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    # Sample random label
    labels = np.arange(0, 10).reshape(-1, 1)

    # Generate images from random noise and label
    gen_imgs = generator.predict([z,labels])
    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(10, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" %labels[cnt])
            cnt += 1
    plt.show()

# Set hyperparameters
iterations = 120000
batch_size = 32
sample_interval = 1000

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)