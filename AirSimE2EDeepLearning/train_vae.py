from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softplus
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape

# model definition class
class VAEModel(Model):
    def __init__(self, n_z, stddev_epsilon=1e-6):
        super(VAEModel, self).__init__()
    
        self.n_z = n_z
        self.stddev_epsilon = stddev_epsilon

        # Encoder architecture
        self.conv1 = Conv2D(input_shape=(28, 28, 1), filters=64, kernel_size=4, strides=2, activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=4, strides=2, activation='relu')
        self.bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.flatten = Flatten()
        self.d1 = Dense(units=1024, activation='relu')
        self.bn2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.d2 = Dense(units=2*self.n_z, activation='relu')

        # Latent space
        self.mean_params = Lambda(lambda x: x[:, :self.n_z])
        self.stddev_params = Lambda(lambda x: x[:, self.n_z:])

        # Decoder architecture
        self.d3 = Dense(units=1024, activation='relu')
        self.bn3 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.d4 = Dense(units=128 * 7 * 7, activation='relu')
        self.bn4 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.reshape = Reshape((7, 7, 128))
        self.deconv1 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.bn5 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.deconv2 = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')

    def call(self, x):

        # Encoding
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn2(x)
        x = self.d2(x)
        means = self.mean_params(x)
        stddev = softplus(self.stddev_params(x)) + self.stddev_epsilon

        # Decoding
        x = means + stddev * random_normal(tf.shape(means))
        x = self.d3(x)
        x = self.bn3(x)
        x = self.d4(x)
        x = self.bn4(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.bn5(x)
        x = self.deconv2(x)

        return x, means, stddev


@tf.function
def compute_loss(y, y_pred, mean, stddev):

    # copute reconstruction loss
    recon_loss = tf.reduce_mean(tf.keras.losses.MSE(y, y_pred))

    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mean) + tf.square(stddev) - tf.math.log(1e-8 + tf.square(stddev)) - 1, [1]))

    return recon_loss, kl_loss

# tf function to train
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions, means, stddev = model(images)
        recon_loss, kl_loss = compute_loss(labels, predictions, means, stddev)
        loss = recon_loss + kl_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_rec_loss(recon_loss)
    train_kl_loss(kl_loss)

# tf function to test
@tf.function
def test(images, labels):
    predictions, means, stddev = model(images)
    recon_loss, kl_loss = compute_loss(labels, predictions, means, stddev)

    test_rec_loss(recon_loss)
    test_kl_loss(kl_loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\cooked_data', type=str)
    parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\models', type=str)
    parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
    parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=20, type=int)
    parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=32, type=int)
    args = parser.parse_args()

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # upload train and test datasets
    train_dataset = h5py.File(os.path.join(args.data_dir, 'train.h5'), 'r')
    train_dataset = np.asarray(train_dataset['image']) / 255.0
    x_train = np.expand_dims(train_dataset[:,0,:,:], axis=1).astype(np.float32)
    y_train = np.expand_dims(train_dataset[:,3,:,:], axis=1).astype(np.float32)
    x_train = x_train.transpose(0, 2, 3, 1) # NCHW => NHWC
    y_train = y_train.transpose(0, 2, 3, 1) # NCHW => NHWC

    test_dataset = h5py.File(os.path.join(args.data_dir, 'test.h5'), 'r')
    test_dataset = np.asarray(test_dataset['image']) / 255.0
    x_test = np.expand_dims(test_dataset[:,0,:,:], axis=1).astype(np.float32)
    y_test = np.expand_dims(test_dataset[:,3,:,:], axis=1).astype(np.float32)
    x_test = x_test.transpose(0, 2, 3, 1) # NCHW => NHWC
    y_test = y_test.transpose(0, 2, 3, 1) # NCHW => NHWC

    # convert to tf format dataset and prepare batches
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)
    
    # create model, loss and optimizer
    model = VAEModel(n_z=args.n_z)
    optimizer = tf.keras.optimizers.Adam()

    # define metrics
    train_rec_loss = tf.keras.metrics.Mean(name='train_rec_loss')
    train_kl_loss = tf.keras.metrics.Mean(name='train_kl_loss')
    test_rec_loss = tf.keras.metrics.Mean(name='test_rec_loss')
    test_kl_loss = tf.keras.metrics.Mean(name='test_kl_loss')
    #metrics_writer = tf.summary.create_file_writer(args.output_dir)

    # check if output folder exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # train
    print('Start training...')
    for epoch in range(args.epochs):

        for images, labels in train_ds:
            train(images, labels)

        for test_images, test_labels in test_ds:
            test(test_images, test_labels)
        
        #with metrics_writer.as_default():
        #    tf.summary.scalar('Train reconstruction loss', train_rec_loss.result(), step=epoch)
        #    tf.summary.scalar('Train KL loss', train_kl_loss.result(), step=epoch)
        #    tf.summary.scalar('Test reconstruction loss', test_rec_loss.result(), step=epoch)
        #    tf.summary.scalar('Test KL loss', test_kl_loss.result(), step=epoch)

        # save model
        if epoch % 5 == 0 and epoch > 0:
            print('Saving weights to {}'.format(args.output_dir))
            model.save_weights(os.path.join(args.output_dir, "vaemodel{}.ckpt".format(epoch)))
        
        print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_rec_loss.result()+train_kl_loss.result(), test_rec_loss.result()+test_kl_loss.result()))