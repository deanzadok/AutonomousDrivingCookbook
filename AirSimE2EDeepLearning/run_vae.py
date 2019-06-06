from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from train_vae import VAEModel
from PIL import Image

# tf function for prediction
@tf.function
def predict_image(image):
    return model(image)

if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\models\\vaemodel30.ckpt', type=str)
    parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
    parser.add_argument('--img', '-img', help='image file path', default='AirSimE2EDeepLearning\\img.png', type=str)
    parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=28, type=int)
    parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present fron camera on screen', action='store_true')
    args = parser.parse_args()

    # Load the model
    model = VAEModel(n_z=args.n_z)
    model.load_weights(args.path)

    # Load image
    im = Image.open(args.img)
    if args.res > 0:
        im = im.resize((args.res, args.res), Image.ANTIALIAS)
    input_img = (np.expand_dims(np.expand_dims(np.array(im), axis=-1), axis=0) / 255.0).astype(np.float32)

    # predict next image
    prediction = predict_image(input_img)

    # present mean values
    print("mean values:")
    print(prediction[1].numpy())
    print("stddev values:")
    print(prediction[2].numpy())

    # present predicted image
    predicted_image = prediction[0].numpy().squeeze(axis=0).squeeze(axis=-1)
    predicted_image = Image.fromarray(np.uint8(predicted_image*255))
    predicted_image.show()
    