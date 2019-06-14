from __future__ import absolute_import, division, print_function, unicode_literals
import os
import h5py
import argparse
import tensorflow as tf
from PIL import Image
from vae_model import VAEModel
from utils import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-model_path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\imitation_4images_vae_28\\vaemodel35.ckpt', type=str)
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\cooked_data\\imitation_to_segmentation_4images_28', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\segmentation_layers_loss\\imitation_to_segmentation_4images_vae_28_1layers', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=40, type=int)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--trainable_layers', '-trainable_layers', help='number of trainable decoding layers', default=1, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=28, type=int)
parser.add_argument('--train_size', '-train_size', help='number of images in the training dataset', default=5000, type=int)
args = parser.parse_args()

# tf function to train
@tf.function
def train(images, labels):
    with tf.GradientTape() as tape:
        predictions, _, _, _ = model(images)
        loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

# tf function to test
@tf.function
def test(images, labels):
    predictions, _, _, _ = model(images)
    loss = tf.reduce_mean(tf.keras.losses.MSE(labels, predictions))

    test_loss(loss)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# get train and test datasets
train_ds, test_ds = dataset.create_dataset(args.data_dir, args.batch_size, args.train_size, label_type='depth')

train_loss_list = []
test_loss_list = []

# prepare boolean vars for decoding layers
trainable_layers = [True]*5
frozen_layers = 5 - args.trainable_layers
for i in range(frozen_layers):
    trainable_layers[i] = False

# create model, loss and optimizer
model = VAEModel(n_z=args.n_z, res=args.res, trainable_encoder=False, trainable_decoder=trainable_layers)
model.load_weights(args.model_path)
optimizer = tf.keras.optimizers.Adam()

# define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

print('Start training...')
for epoch in range(args.epochs):

    for images, labels in train_ds:
        train(images, labels)

    for test_images, test_labels in test_ds:
        test(test_images, test_labels)
    
    with metrics_writer.as_default():
        tf.summary.scalar('Train loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Test loss', test_loss.result(), step=epoch)
    train_loss_list.append(train_loss.result())
    test_loss_list.append(test_loss.result())

    # save model
    if epoch % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "depthmodel{}.ckpt".format(epoch)))
    
    print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss.result(), test_loss.result()))

#plot_graph(args.epochs, train_loss_list, test_loss_list, args.output_dir)
