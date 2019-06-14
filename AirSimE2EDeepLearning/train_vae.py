from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from vae_model import VAEModel
from utils import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\cooked_data\\imitation_4images', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\test', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=40, type=int)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
args = parser.parse_args()

@tf.function
def compute_loss(y, y_pred, means, stddev):

    # copute reconstruction loss
    recon_loss = loss_object(y, y_pred)

    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    #kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(means) + tf.square(stddev) - tf.math.log(1e-8 + tf.square(stddev)) - 1, [1]))
    kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum((1+stddev-tf.math.pow(means, 2)-tf.math.exp(stddev)), axis=1))

    return recon_loss, kl_loss

# tf function to train
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions, means, stddev, _ = model(images)
        recon_loss, kl_loss = compute_loss(labels, predictions, means, stddev)
        loss = recon_loss + kl_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_rec_loss(recon_loss)
    train_kl_loss(kl_loss)

# tf function to test
@tf.function
def test(images, labels):
    predictions, means, stddev, _ = model(images)
    recon_loss, kl_loss = compute_loss(labels, predictions, means, stddev)

    test_rec_loss(recon_loss)
    test_kl_loss(kl_loss)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# get train and test datasets
#train_ds, test_ds = create_dataset_from_folder(args.data_dir, args.batch_size, args.res)
train_ds, test_ds = dataset.create_dataset(args.data_dir, args.batch_size, label_type='image')

# create model, loss and optimizer
model = VAEModel(n_z=args.n_z, res=args.res)
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# define metrics
train_rec_loss = tf.keras.metrics.Mean(name='train_rec_loss')
train_kl_loss = tf.keras.metrics.Mean(name='train_kl_loss')
test_rec_loss = tf.keras.metrics.Mean(name='test_rec_loss')
test_kl_loss = tf.keras.metrics.Mean(name='test_kl_loss')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

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
    
    with metrics_writer.as_default():
        tf.summary.scalar('Train reconstruction loss', train_rec_loss.result(), step=epoch)
        tf.summary.scalar('Train KL loss', train_kl_loss.result(), step=epoch)
        tf.summary.scalar('Test reconstruction loss', test_rec_loss.result(), step=epoch)
        tf.summary.scalar('Test KL loss', test_kl_loss.result(), step=epoch)

    # save model
    if epoch % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "vaemodel{}.ckpt".format(epoch)))
    
    print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_rec_loss.result()+train_kl_loss.result(), test_rec_loss.result()+test_kl_loss.result()))