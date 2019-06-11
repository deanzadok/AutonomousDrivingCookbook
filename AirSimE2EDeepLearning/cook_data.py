#%matplotlib inline
import numpy as np
import pandas as pd
import h5py
from matplotlib import use
use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import Cooking
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\imitation_4images', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to destination folder', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Data\\cooked_data\\imitation_4images_64', type=str)
parser.add_argument('--buffer_size', '-buffer_size', help='number of images in one sample', default=2, type=int)
parser.add_argument('--images_gap', '-images_gap', help='number of images in one sample', default=3, type=int)
parser.add_argument('--img_type', '-img_type', help='type of image from [rgb, depth], depth and grayscale are the same', default='depth', type=str)
parser.add_argument('--label_type', '-label_type', help='type of image from [action, depth]', default='action', type=str)
parser.add_argument('--dest_res', '-dest_res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)

args = parser.parse_args()
# No test set needed, since testing in our case is running the model on an unseen map in AirSim
train_test_split = [0.9, 0.1]

# list of all recordings data_folders
data_folders = [name for name in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, name))]
data_folders = [os.path.join(args.data_dir, f) for f in data_folders]

Cooking.cook(data_folders, args.output_dir, args.buffer_size, args.images_gap, args.img_type, args.dest_res, args.label_type, train_test_split)