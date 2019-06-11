from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from vae_model import VAEModel
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\imitation_4images_vae_64\\vaemodel35.ckpt', type=str)
parser.add_argument('--mat_path', '-mat_path', help='model file path', default='C:\\Users\\t-dezado\\Downloads\\nyu_depth_v2_labeled.mat', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--img', '-img', help='image file path', default='AirSimE2EDeepLearning\\img.png', type=str)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
parser.add_argument('--show', '-show', help='choose what to do from [predict, inter]', default='inter', type=str)
args = parser.parse_args()

# tf function for prediction
@tf.function
def predict_image(image, inter=None):
    return model(image, inter)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

"""
dataset_dict = h5py.File(args.mat_path, 'r')
images_dataset = np.array(dataset_dict['images']).transpose(0,2,3,1).astype(np.float32) / 255.0
images_dataset = np.expand_dims(0.2989 * images_dataset[:,:,:,0] + 0.5870 * images_dataset[:,:,:,1] + 0.1140 * images_dataset[:,:,:,2], axis=-1)
depths_dataset = np.expand_dims(np.array(dataset_dict['depths']), axis=-1).astype(np.float32) / 10.0
labels_dataset = np.expand_dims(np.array(dataset_dict['labels']), axis=-1).astype(np.float32) / 255.0
"""

# Load the model
model = VAEModel(n_z=args.n_z, res=args.res)
model.load_weights(args.path)

# Load image
im = Image.open(args.img)
if args.res > 0:
    im = im.resize((args.res, args.res), Image.ANTIALIAS)
input_img = (np.expand_dims(np.expand_dims(np.array(im), axis=-1), axis=0) / 255.0).astype(np.float32)

"""
im = Image.fromarray(images_dataset[350].squeeze(axis=-1)).resize((args.res,args.res),Image.ANTIALIAS)
input_img = np.expand_dims(np.expand_dims(np.array(im),axis=0),axis=-1)
"""

if args.show == 'inter': # interpolate over the latent space

    latent_array = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

    output_im = np.zeros((args.res*latent_array.shape[0],args.res*args.n_z))

    for i in range(args.n_z):

        for j in range(latent_array.shape[0]):
            # predict next image
            latent_vector = np.zeros((1, args.n_z)).astype(np.float32)
            latent_vector[0][i] = latent_array[j]
            prediction, _, _, _ = predict_image(input_img, inter=latent_vector)

            # add predicted image to sequence
            predicted_image = prediction.numpy().squeeze(axis=0).squeeze(axis=-1)*255
            output_im[j*args.res:(j+1)*args.res,i*args.res:(i+1)*args.res] = predicted_image

    # present sequence of images
    predicted_image = Image.fromarray(np.uint8(output_im))
    predicted_image.show()

else:

    # predict next image
    prediction, _, _, z = predict_image(input_img)

    # present mean values
    print("z values:")
    print(z.numpy())

    # present predicted image
    predicted_image = prediction.numpy().squeeze(axis=0).squeeze(axis=-1)
    predicted_image = Image.fromarray(np.uint8(predicted_image*255))
    predicted_image.show()
