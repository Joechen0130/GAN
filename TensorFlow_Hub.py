import matplotlib.pyplot as plt
import tensorflow as tf
import  tensorflow_hub as hub

with tf.Graph().as_default():
    module = hub.Module("https://tfhub.dev/google/progan-128/1")

    latent_dim = 512
    latent_vector = tf.random_normal([1, latent_dim], seed=1337)

    interpolted_images = module(latent_vector)

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        image_out = session.run(interpolted_images)
plt.imshow(image_out.reshape(128,128,3))
plt.show()