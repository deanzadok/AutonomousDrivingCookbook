import setup_path
import airsim
import argparse
import numpy as np
import time
import sys
import json
import datetime
from coverage_map import CoverageMap, HistoryMap
import os
import PIL

from cntk_agent import DeepQAgent

#from cntk.initializer import he_uniform, normal
#from cntk.layers import Sequential, Convolution2D, Dense, default_options, Activation, MaxPooling, Dense, Dropout, For
#from cntk.layers.typing import Signature, Tensor
#from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square, placeholder, minus, constant

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\Desktop\\model17260000', type=str)
parser.add_argument('--type', '-type', help='experiment type from [regular, with_rgb]', default='with_rgb', type=str)
parser.add_argument('--state_size', '-state_size', help='the size of the state of the coverage map', default=4000, type=int)
parser.add_argument('--reward_norm', '-reward_norm', help='factor to normalize the reward', default=2.0, type=float)
args = parser.parse_args()

nb_actions = 5
input_shape = (4, 84, 84)

# load agent model
agent = DeepQAgent(input_shape, nb_actions, monitor=True)
agent.load(args.path)

print('Connecting to AirSim...')
car_client = airsim.CarClient()
car_client.confirmConnection()
car_client.enableApiControl(True)
car_controls = airsim.CarControls()
print('Connected!')

# initiate coverage map
"""
start_point = [840.0, 1200.0, 32.0]
coverage_map = CoverageMap(start_point=start_point, map_size=12000, scale_ratio=1, state_size=4000, input_size=84, height_threshold=0.9, reward_norm=3000.0)
coverage_map.set_client(car_client)
"""
start_point = [-1200.0, -500.0, 62.000687]
map_boundaries = [[-1400,400],[-1400,400]]
hisMap = HistoryMap(start_point=start_point, map_size=19, input_size=84, map_boundaries=map_boundaries)
hisMap.set_client(client=car_client)

def interpret_action(action):
    car_controls.brake = 0
    car_controls.throttle = 0.5
    if action == 0:
        car_controls.steering = 0
    elif action == 1:
        car_controls.steering = 0.5
    elif action == 2:
        car_controls.steering = -0.5
    elif action == 3:
        car_controls.steering = 0.25
    else:
        car_controls.steering = -0.25
    return car_controls

# Gets a coverage image from AirSim
def get_cov_image():

    state, reward = hisMap.get_state()
    #state = coverage_map.get_map_scaled()

    # debug only
    #print(reward)
    im = PIL.Image.fromarray(np.uint8(state))
    im.save("DistributedRL\\debug\\{}.png".format(time.time()))

    state = state / 255.0

    return state

# Gets an image from AirSim
def get_image():
    image_response = car_client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    if image1d.size > 1:
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

        # debug only
        #im = PIL.Image.fromarray(np.uint8(image_rgba))
        #im.save("DistributedRL\\debug\\{}.png".format(time.time()))

        image_rgba = image_rgba / 255.0
        return image_rgba[60:144,86:170,0:3].astype(float)
    
    return np.zeros((84,84,3)).astype(float)

def get_depth_image():

    responses = car_client.simGetImages([airsim.ImageRequest("RCCamera", airsim.ImageType.DepthPerspective, True, False)])
    img1d = np.array(responses[0].image_data_float, dtype=np.float)

    if img1d.size > 1:

        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = PIL.Image.fromarray(img2d)

        # debug only
        #image_png = image.convert('RGB')
        #image_png.save("DistributedRL\\debug\\{}.png".format(time.time()))

        im_final = np.array(image.resize((84, 84)).convert('L')) 
        #im_final = im_final / 255.0

        return im_final

    return np.zeros((84,84)).astype(float)

print('Running car for a few seconds...')
car_controls.steering = 0
car_controls.throttle = 0.4
car_controls.brake = 0
car_client.setCarControls(car_controls)
time.sleep(0.5)

print('Running model')
image = get_cov_image()
#image = get_depth_image()

while(True):

    action, qvalues = agent.act(image, eval=True)
    car_controls = interpret_action(action)
    car_client.setCarControls(car_controls)

    #print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(next_state, car_controls.steering, car_controls.throttle, car_controls.brake))
    #print(qvalues)

    image = get_cov_image()
    #image = get_depth_image()