import tensorflow as tf

class Generator:
    def __init__(self):
        pass

    def __call__(self):
        # with tf.variable_scope('generator', reuse=True):
        pass


class Discriminator:
    def __init__(self, filters = [64, 128, 256, 512]):
        self.filters = filters
    
    def __call__(self, input, is_train):
        with tf.variable_scope('discriminator'):
            convolution_1 = tf.layers.conv2d(input, self.filters[0], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_1 = tf.layers.batch_normalization(convolution_1, training=is_train, epsilon=1e-5, decay = 0.9)
            activation_1 = tf.nn.leaky_relu(batch_norm_1) #alpha = 0.2

            convolution_2 = tf.layers.conv2d(activation_1, self.filters[1], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_2 = tf.layers.batch_normalization(convolution_2, training=is_train, epsilon=1e-5, decay = 0.9)
            activation_2 = tf.nn.leaky_relu(batch_norm_2)

            convolution_3 = tf.layers.conv2d(activation_2, self.filters[2], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_3 = tf.layers.batch_normalization(convolution_3, training=is_train, epsilon=1e-5, decay = 0.9)
            activation_3 = tf.nn.leaky_relu(batch_norm_3)

            convolution_4 = tf.layers.conv2d(activation_3, self.filters[3], kernel_size=[5,5], strides=2, padding="SAME")
            batch_norm_4 = tf.layers.batch_normalization(convolution_4, training=is_train, epsilon=1e-5, decay = 0.9)
            activation_4 = tf.nn.leaky_relu(batch_norm_4)

            batch_size = activation_4.get_shape()[0].value
            reshaped = tf.reshape(activation_4, shape=[batch_size, -1])
            dense = tf.layers.dense(reshaped, 2)

            return dense




