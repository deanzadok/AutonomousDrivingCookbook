from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# model definition class
class RLModel(Model):
  def __init__(self, num_actions=5):
    super(RLModel, self).__init__()
    self.conv1 = Conv2D(filters=16, kernel_size=8, strides=4, activation='relu')
    self.conv2 = Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')
    self.conv3 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(units=256, activation='relu')
    self.d2 = Dense(units=num_actions, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# tf function to train
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# tf function to test
@tf.function
def test(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\cooked_data', type=str)
    parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\models', type=str)
    parser.add_argument('--num_actions', '-num_actions', help='number of actions for the model to perdict', default=5, type=int)
    parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
    parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=20, type=int)
    args = parser.parse_args()

    # upload train and test datasets
    train_dataset = h5py.File(os.path.join(args.data_dir, 'train.h5'), 'r')
    x_train = np.asarray(train_dataset['image']) / 255.0
    y_train = np.asarray(train_dataset['label']).astype(int)
    x_train = x_train.transpose(0, 2, 3, 1) # NCHW => NHWC

    test_dataset = h5py.File(os.path.join(args.data_dir, 'test.h5'), 'r')
    x_test = np.asarray(test_dataset['image']) / 255.0
    y_test = np.asarray(test_dataset['label']).astype(int)
    x_test = x_test.transpose(0, 2, 3, 1) # NCHW => NHWC

    # convert to tf format dataset and prepare batches
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create model, loss and optimizer
    model = RLModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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
        
        # save model
        if epoch % 5 == 0 and epoch > 0:
            print('Saving weights to {}'.format(args.output_dir))
            model.save_weights(os.path.join(args.output_dir, "model{}.ckpt".format(epoch)))
        
        print('Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))