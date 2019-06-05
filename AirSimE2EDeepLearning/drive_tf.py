import sys
sys.path.append('C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\Github\\AutonomousDrivingCookbook\\DistributedRL\\Share\\scripts_downpour\\app')
import setup_path
import airsim
import os
import numpy as np
import argparse
import tensorflow as tf
from train import RLModel
from PIL import Image
from coverage_map import CoverageMap
import cv2
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Desktop\\model15.ckpt', type=str)
parser.add_argument('--camera', '-camera', help='type of the camera. choose from [rgb, depth, grayscale]', default='grayscale', type=str)
parser.add_argument('--debug', '-debug', dest='debug', help='debug mode, present fron camera on screen', action='store_true')
parser.add_argument('--store', '-store', dest='store', help='store images to experiment folder', action='store_true')
args = parser.parse_args()

# tf function to test
@tf.function
def predict_action(image):
    return model(image)

if __name__ == "__main__":

    # allow growth is possible using an env var in tf2.0
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    if args.store:
        # create experiments directories
        experiment_dir = os.path.join('C:\\Users\\t-dezado\\OneDrive - Microsoft\\Documents\\AirSim', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        images_dir = os.path.join(experiment_dir, 'images')
        os.makedirs(images_dir)
        # create txt file
        airsim_rec = open(os.path.join(experiment_dir,"airsim_rec.txt"),"w") 
        airsim_rec.write("TimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tRPM\tSpeed\tSteering\tImageFile\n") 

    # Load the model
    model = RLModel()
    model.load_weights(args.path)

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # let the car start driving
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)

    # create coverage map and connect to client
    #start_point = [-290.0, 10050.0, 10.0]
    start_point = [0.0, 0.0, 0.0]
    covMap = CoverageMap(start_point=start_point, map_size=64000, scale_ratio=20, state_size=6000, input_size=20, height_threshold=0.9, reward_norm=30, paint_radius=15)
    covMap.set_client(client=client)

    # actions list
    actions = [-1.0, -0.5, 0, 0.5, 1.0]

    # initiate buffer
    buffer = np.zeros((4, 84, 84), dtype=np.float32)

    print('Running model')

    while(True):

        if args.camera == 'depth':
            # get depth image from airsim
            responses = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])
            img1d = np.array(responses[0].image_data_float, dtype=np.float)
            img1d = 255/np.maximum(np.ones(img1d.size), img1d)
            if img1d.size > 1:
                img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
                image = Image.fromarray(img2d)
                image_np = np.array(image.resize((84, 84)).convert('L')) 
            else:
                image_np = np.zeros((84,84)).astype(float)
        else: # args.camera = 'rgb' or 'grayscale'
            # get image from AirSim
            image_response = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)])[0]
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
            if img1d.size > 1:
                img2d = np.reshape(img1d, (image_response.height, image_response.width, 3))
                image = Image.fromarray(img2d)
                if args.camera == 'grayscale':
                    image = image.convert('L')
                image_np = np.array(image.resize((84, 84)))
                
            else:
                if args.camera == 'grayscale':
                    image_np = np.zeros((84,84)).astype(float)
                else:
                    image_np = np.zeros((84,84,3)).astype(float)

        # get coverage image
        cov_image, reward = covMap.get_state_from_pose()
        #print("reward: {}".format(reward))

        # combine both inputs
        if args.camera == 'depth' or args.camera == 'grayscale':
            image_np[:cov_image.shape[0],:cov_image.shape[1]] = cov_image
        else:
            cov_rgb_image = np.expand_dims(cov_image, axis=2)
            cov_rgb_image = np.repeat(cov_rgb_image, 3, axis=2)
            image_np[:cov_rgb_image.shape[0],:cov_rgb_image.shape[1],:] = cov_rgb_image

        # present state image if debug mode is on
        if args.debug:
            cv2.imshow('navigation map', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # append to buffer
        image_np = image_np / 255.0
        buffer[:-1] = buffer[1:]
        buffer[-1] = image_np
        
        # convert input to [1,84,84,4]
        input_img = np.expand_dims(buffer, axis=0).transpose(0, 2, 3, 1).astype(np.float64)

        # predict steering action
        predictions = predict_action(input_img)
        car_controls.steering = actions[np.argmax(predictions)]

        client.setCarControls(car_controls)

        print('steering = {0}, qvalues = {1}'.format(car_controls.steering, predictions))