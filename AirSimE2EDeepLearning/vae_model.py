import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape

# model definition class
class VAEModel(Model):
    def __init__(self, n_z, stddev_epsilon=1e-6, final_activation='sigmoid', trainable_encoder=True, trainable_decoder=[True]*7, res=28):
        super(VAEModel, self).__init__()
    
        self.n_z = n_z
        self.stddev_epsilon = stddev_epsilon
        self.res = res

        # Encoder architecture
        self.conv1 = Conv2D(filters=64, kernel_size=4, strides=2, trainable=trainable_encoder)
        self.conv2 = Conv2D(filters=128, kernel_size=4, strides=2, trainable=trainable_encoder)
        if self.res == 64:
            self.conv3 = Conv2D(filters=256, kernel_size=4, strides=2, trainable=trainable_encoder)
        self.bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=trainable_encoder)
        self.flatten = Flatten()
        self.d1 = Dense(units=1024, trainable=trainable_encoder)
        self.bn2 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=trainable_encoder)
        self.d2 = Dense(units=2*self.n_z, trainable=trainable_encoder)

        # Latent space
        self.mean_params = Lambda(lambda x: x[:, :self.n_z])
        self.stddev_params = Lambda(lambda x: x[:, self.n_z:])

        # Decoder architecture
        self.d3 = Dense(units=1024, trainable=trainable_decoder[0])
        self.bn3 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=trainable_decoder[1])
        self.d4 = Dense(units=128 * 7 * 7, trainable=trainable_decoder[2])
        self.bn4 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=trainable_decoder[3])
        self.reshape = Reshape((7, 7, 128))
        if self.res == 64:
            self.deconv1 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='valid', trainable=trainable_decoder[4])
            self.bn5 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=trainable_decoder[5])

        self.deconv2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', trainable=trainable_decoder[4])
        self.bn6 = BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=trainable_decoder[5])
        self.deconv3 = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation=final_activation, trainable=trainable_decoder[6])

        #x = np.random.normal(size=(1, 28, 28, 1))
        #x = tf.convert_to_tensor(x)
        #_ = self.call(x)

    def call(self, x, inter=None):

        # Encoding
        x = relu(self.conv1(x))
        x = self.conv2(x)
        if self.res == 64:
            x = self.conv3(relu(x))
        x = relu(self.bn1(x))
        x = self.flatten(x)
        x = relu(self.bn2(self.d1(x)))
        x = self.d2(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5*self.stddev_params(x))
        eps = random_normal(tf.shape(stddev))

        # Decoding
        z = means + eps * stddev
        if inter is not None:
            z = tf.keras.layers.add([z,inter])
        x = relu(self.bn3(self.d3(z)))
        x = relu(self.bn4(self.d4(x)))
        x = self.reshape(x)
        if self.res == 64:
            x = relu(self.bn5(self.deconv1(x)))
        x = relu(self.bn6(self.deconv2(x)))
        x = self.deconv3(x)

        return x, means, stddev, z
