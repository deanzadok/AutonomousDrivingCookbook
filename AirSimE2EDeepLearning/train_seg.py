from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from train_vae import VAEModel
from PIL import Image

# tf function to train
@tf.function
def train(images, labels):
    if args.res > 0:
        images = tf.image.resize(images, (args.res, args.res))
        labels = tf.image.resize(labels, (args.res, args.res))
    with tf.GradientTape() as tape:
        predictions, _, _ = model(images)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

# tf function to test
@tf.function
def test(images, labels):
    if args.res > 0:
        images = tf.image.resize(images, (args.res, args.res))
        labels = tf.image.resize(labels, (args.res, args.res))
    predictions, _, _ = model(images)
    loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))

    test_loss(loss)

if __name__ == "__main__":
    
    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-data', help='model file path', default='C:\\Users\\t-dezado\\Downloads\\nyu_depth_v2_labeled.mat', type=str)
    parser.add_argument('--model_path', '-model_path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\models\\vaemodel35.ckpt', type=str)
    parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\models', type=str)
    parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
    parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=40, type=int)
    parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
    parser.add_argument('--img', '-img', help='image file path', default='AirSimE2EDeepLearning\\img.png', type=str)
    parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=28, type=int)
    parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present fron camera on screen', action='store_true')
    args = parser.parse_args()

    dataset_dict = h5py.File(args.data, 'r')
    images_dataset = np.array(dataset_dict['images']).transpose(0,2,3,1).astype(np.float32) / 255.0
    images_dataset = np.expand_dims(0.2989 * images_dataset[:,:,:,0] + 0.5870 * images_dataset[:,:,:,1] + 0.1140 * images_dataset[:,:,:,2], axis=-1)
    depths_dataset = np.expand_dims(np.array(dataset_dict['depths']), axis=-1).astype(np.float32) / 10.0
    train_split = int(0.9 * depths_dataset.shape[0])

    x_train = images_dataset[:train_split,:,:,:]
    y_train = depths_dataset[:train_split,:,:,:]
    x_test = images_dataset[train_split:,:,:,:]
    y_test = depths_dataset[train_split:,:,:,:]
    
    # convert to tf format dataset and prepare batches
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)
    
    # create model, loss and optimizer
    model = VAEModel(n_z=args.n_z)
    model.load_weights(args.model_path)
    optimizer = tf.keras.optimizers.Adam()

    # define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
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
            model.save_weights(os.path.join(args.output_dir, "segmodel{}.ckpt".format(epoch)))
        
        print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss.result(), test_loss.result()))