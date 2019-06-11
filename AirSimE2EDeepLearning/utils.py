import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def create_dataset(data_dir, batch_size, label_type='image'):

    # get train dataset 
    train_dataset = h5py.File(os.path.join(data_dir, 'train.h5'), 'r')
    train_images_dataset = np.asarray(train_dataset['image']) / 255.0
    train_labels_dataset = np.asarray(train_dataset['label']) / 255.0

    # get test dataset 
    test_dataset = h5py.File(os.path.join(data_dir, 'test.h5'), 'r')
    test_images_dataset = np.asarray(test_dataset['image']) / 255.0
    test_labels_dataset = np.asarray(test_dataset['label']) / 255.0

    # get only first image from the input sequence
    x_train = np.expand_dims(train_images_dataset[:,0,:,:], axis=1)
    x_test = np.expand_dims(test_images_dataset[:,0,:,:], axis=1)
    if label_type == 'image':
        y_train = np.expand_dims(train_images_dataset[:,-1,:,:], axis=1)
        y_test = np.expand_dims(test_images_dataset[:,-1,:,:], axis=1)
    else:
        y_train = train_labels_dataset
        y_test = test_labels_dataset

    # convert data format
    x_train = x_train.transpose(0, 2, 3, 1) # NCHW => NHWC
    y_train = y_train.transpose(0, 2, 3, 1) # NCHW => NHWC
    x_test = x_test.transpose(0, 2, 3, 1) # NCHW => NHWC
    y_test = y_test.transpose(0, 2, 3, 1) # NCHW => NHWC

    # convert data type to float32
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # convert to tf format dataset and prepare batches
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, test_ds

def plot_graph(length, train_loss_list, test_loss_list, output_dir):

    # plot the results
    plt.plot(range(length), train_loss_list)
    plt.plot(range(length), test_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))