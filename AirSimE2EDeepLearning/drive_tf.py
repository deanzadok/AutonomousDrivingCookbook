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

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\OneDrive - Microsoft\\Desktop\\model25.ckpt', type=str)
args = parser.parse_args()

# Gets a coverage image from AirSim
def get_cov_image(covmap):

    state, reward = covmap.get_state_from_pose()

    # debug only
    #print(reward)
    #im = PIL.Image.fromarray(np.uint8(state))
    #im.save("DistributedRL\\debug\\{}.png".format(time.time()))

    state = state / 255.0
    return state


def get_depth_image(client):

    responses = client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])
    img1d = np.array(responses[0].image_data_float, dtype=np.float)

    if img1d.size > 1:

        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)

        # debug only
        #image_png = image.convert('RGB')
        #image_png.save("DistributedRL\\debug\\{}.png".format(time.time()))

        im_final = np.array(image.resize((84, 84)).convert('L')) 
        im_final = im_final / 255.0
        return im_final

    return np.zeros((84,84)).astype(float)

# tf function to test
@tf.function
def predict_action(image):
    return model(image)

if __name__ == "__main__":

    # Load the model
    model = RLModel()
    model.load_weights(args.path)

    # connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # let the car start driving
    car_controls.throttle = 0.3
    car_controls.steering = 0
    client.setCarControls(car_controls)

    # create coverage map and connect to client
    start_point = [-290.0, 10050.0, 10.0]
    covMap = CoverageMap(start_point=start_point, map_size=32000, scale_ratio=20, state_size=6000, input_size=20, height_threshold=0.9, reward_norm=30, paint_radius=15)
    covMap.set_client(client=client)

    # actions list
    actions = [-1.0, -0.5, 0, 0.5, 1.0]

    # initiate buffer
    buffer = np.zeros((4, 84, 84), dtype=np.float32)

    print('Running model')

    while(True):

        # combine both inputs
        cov_image = get_cov_image(covMap)
        image = get_depth_image(client)
        image[:cov_image.shape[0],:cov_image.shape[1]] = cov_image

        # append to buffer
        buffer[:-1] = buffer[1:]
        buffer[-1] = image
        
        # convert input to [1,84,84,4]
        input_img = np.expand_dims(buffer, axis=0).transpose(0, 2, 3, 1).astype(np.float64)

        # predict steering action
        predictions = predict_action(input_img)
        car_controls.steering = actions[np.argmax(predictions)]

        client.setCarControls(car_controls)

        print('steering = {0}, qvalues = {1}'.format(car_controls.steering, predictions))