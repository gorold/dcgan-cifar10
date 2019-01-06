import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave

class Discriminator:
    def __init__(self, filters = [64, 128, 256, 512]):
        self.filters = filters
    
    def __call__(self, input, is_train, batch_size, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            convolution_1 = tf.layers.conv2d(input, self.filters[0], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_1 = tf.layers.batch_normalization(convolution_1, training=is_train, epsilon=1e-5)
            activation_1 = tf.nn.leaky_relu(batch_norm_1) #alpha = 0.2

            convolution_2 = tf.layers.conv2d(activation_1, self.filters[1], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_2 = tf.layers.batch_normalization(convolution_2, training=is_train, epsilon=1e-5)
            activation_2 = tf.nn.leaky_relu(batch_norm_2)

            convolution_3 = tf.layers.conv2d(activation_2, self.filters[2], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_3 = tf.layers.batch_normalization(convolution_3, training=is_train, epsilon=1e-5)
            activation_3 = tf.nn.leaky_relu(batch_norm_3)

            convolution_4 = tf.layers.conv2d(activation_3, self.filters[3], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_4 = tf.layers.batch_normalization(convolution_4, training=is_train, epsilon=1e-5)
            activation_4 = tf.nn.leaky_relu(batch_norm_4)

            reshaped = tf.reshape(activation_4, shape=[batch_size, 2048])
            out_logit = tf.layers.dense(reshaped, 1, reuse=reuse)
            out = tf.nn.sigmoid(out_logit)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            return out_logit, out

class Generator:
    def __init__(self, filters=[512, 256, 128, 64, 3], size=2):
        self.filters = filters
        self.size = size

    def __call__(self, inputs, is_train=False, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            dense = tf.layers.dense(inputs, self.filters[0] * self.size * self.size)
            
            deconvolution_1 = tf.reshape(dense, shape=[-1, self.size, self.size, self.filters[0]])
            batch_norm_1 = tf.layers.batch_normalization(deconvolution_1, training=is_train, epsilon=1e-5)
            activation_1 = tf.nn.leaky_relu(batch_norm_1) #alpha = 0.2

            deconvolution_2 = tf.layers.conv2d_transpose(activation_1, self.filters[1], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_2 = tf.layers.batch_normalization(deconvolution_2, training=is_train, epsilon=1e-5)
            activation_2 = tf.nn.leaky_relu(batch_norm_2)

            deconvolution_3 = tf.layers.conv2d_transpose(activation_2, self.filters[2], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_3 = tf.layers.batch_normalization(deconvolution_3, training=is_train, epsilon=1e-5)
            activation_3 = tf.nn.leaky_relu(batch_norm_3)

            deconvolution_4 = tf.layers.conv2d_transpose(activation_3, self.filters[3], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_4 = tf.layers.batch_normalization(deconvolution_4, training=is_train, epsilon=1e-5)
            activation_4 = tf.nn.leaky_relu(batch_norm_4)

            deconvolution_5 = tf.layers.conv2d_transpose(activation_4, self.filters[4], kernel_size=[5,5], strides=2, padding="SAME")
            activation_5 = tf.nn.tanh(deconvolution_5)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            return activation_5

class DCGAN:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

    def train(self, train_images, epochs=50, batch_size=32, learning_rate=0.0002, beta1=0.5):

        train_noise_dim = 100
        iterations = train_images.shape[0] // batch_size

        with tf.variable_scope('input'):
            real_image = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="real_image")
            random_input = tf.placeholder(tf.float32, shape=[None, train_noise_dim], name="random_input")
            random_input2 = tf.placeholder(tf.float32, shape=[None, train_noise_dim], name="random_input2")

        fake_image = self.generator(random_input, is_train=True)
        gen_image = self.generator(random_input2, is_train=False, reuse=True)

        real_logit, real = self.discriminator(real_image, batch_size=batch_size, is_train=True)
        fake_logit, fake = self.discriminator(fake_image, batch_size=batch_size, is_train=True, reuse=True)

        discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real)))
        discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake)))
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake)))

        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(discriminator_loss, var_list=self.discriminator.variables)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(generator_loss, var_list=self.generator.variables)
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.discriminator.variables]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                print("Current epoch: {}".format(epoch))
                batch = 0
                train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, train_noise_dim]).astype(np.float)

                for iteration in range(iterations):
                    x_image = train_images[batch*batch_size:(batch+1)*batch_size]
                    batch = batch + 1

                    sess.run(d_clip)
                    _, d_loss = sess.run([discriminator_optimizer, discriminator_loss], feed_dict={real_image: x_image, random_input: train_noise})
                    _, g_loss = sess.run([generator_optimizer, generator_loss], feed_dict={random_input: train_noise})

                    print("Iter: {:d}, D_loss: {:.4f}, G_loss: {:.4f}".format(iteration, d_loss, g_loss))

                    if iteration % 20 == 0:
                        sample = sess.run(gen_image, feed_dict={random_input2: train_noise})
                        sample *= 255
                        location = os.getcwd() + "/generated_images/epoch_{}_iter_{}.jpg".format(epoch, iteration)
                        imsave(location, sample[0])                

                # if epoch % 50 == 0:

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    y_train_index_cars = [i for i, v in enumerate(y_train) if v == 1]
    x_train = np.asarray([v for i, v in enumerate(x_train) if i in y_train_index_cars])
    x_train /= 255
    dcgan = DCGAN()
    dcgan.train(x_train, epochs=30)