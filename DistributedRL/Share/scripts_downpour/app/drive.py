from rl_model import RlModel
import setup_path
import airsim
import argparse
import numpy as np
import time
import sys
import json
import datetime
from coverage_map import CoverageMap
import os
import PIL

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\Desktop\\750171.json', type=str)
parser.add_argument('--type', '-type', help='experiment type from [regular, with_rgb]', default='with_rgb', type=str)
parser.add_argument('--state_size', '-state_size', help='the size of the state of the coverage map', default=4000, type=int)
parser.add_argument('--reward_norm', '-reward_norm', help='factor to normalize the reward', default=2000.0, type=float)
args = parser.parse_args()

buffer_len = 4
if args.type == 'with_rgb':
    buffer_len = 3

model = RlModel(weights_path=None, train_conv_layers=False, exp_type=args.type, buffer_len=buffer_len)
with open(args.path, 'r') as f:
    checkpoint_data = json.loads(f.read())
    model.from_packet(checkpoint_data['model'])

# initiate coverage map
start_point = [500.0, 850.0, 32.0]
coverage_map = CoverageMap(start_point=start_point, map_size=12000, scale_ratio=1, state_size=args.state_size, input_size=84, height_threshold=0.95, reward_norm=args.reward_norm)

print('Connecting to AirSim...')
car_client = airsim.CarClient()
car_client.confirmConnection()
car_client.enableApiControl(True)
car_controls = airsim.CarControls()
coverage_map.set_client(client=car_client)
print('Connected!')

def append_to_ring_buffer(item, rgb_item, buffer, rgb_buffer, buffer_size):
    if (len(buffer) >= buffer_size):
        buffer = buffer[1:]
        rgb_buffer = rgb_buffer[1:]
    buffer.append(item)
    rgb_buffer.append(rgb_item)
    return buffer, rgb_buffer

# Gets a coverage image from AirSim
def get_cov_image():

    state, reward = coverage_map.get_state()

    # debug only
    #print(reward)
    #im = PIL.Image.fromarray(np.uint8(state))
    #im.save("DistributedRL\\debug\\{}.png".format(time.time()))

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

state_buffer = []
rgb_buffer = []

print('Running car for a few seconds...')
car_controls.steering = 0
car_controls.throttle = 0.4
car_controls.brake = 0
car_client.setCarControls(car_controls)
stop_run_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
while(datetime.datetime.now() < stop_run_time):
    time.sleep(0.025)
    cov_image = get_cov_image()
    rgb_image = None
    if args.type == 'with_rgb':
        rgb_image = get_image()
    state_buffer, rgb_buffer = append_to_ring_buffer(cov_image, rgb_image, state_buffer, rgb_buffer, buffer_len)

# slow down a bit
car_controls.throttle = 0.3
car_client.setCarControls(car_controls)

print('Running model')
while(True):
    cov_image = get_cov_image()
    rgb_image = None
    if args.type == 'with_rgb':
        rgb_image = get_image()
    state_buffer, rgb_buffer = append_to_ring_buffer(cov_image, rgb_image, state_buffer, rgb_buffer, buffer_len)
    next_state, dummy, qvalues = model.predict_state(state_buffer, rgb_buffer)

    # Convert the selected state to a control signal
    next_steering, is_reverse, next_brake = model.state_to_control_signals(next_state, car_client.getCarState())

    # Take the action
    car_controls.steering = next_steering
    car_controls.brake = next_brake
    if is_reverse:
        car_controls.throttle = -0.3
        car_controls.is_manual_gear = True
        car_controls.manual_gear = -1
    else:
        car_controls.throttle = 0.3
        car_controls.is_manual_gear = False
        car_controls.manual_gear = 0

    #print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(next_state, car_controls.steering, car_controls.throttle, car_controls.brake))
    print(qvalues)
    car_client.setCarControls(car_controls)

    time.sleep(0.01)
