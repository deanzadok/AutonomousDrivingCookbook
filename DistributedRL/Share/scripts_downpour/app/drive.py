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

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='C:\\Users\\t-dezado\\Desktop\\1288868.json', type=str)
args = parser.parse_args()

#old_path = os.path.join(os.getcwd(),"DistributedRL\\Share\\checkpoint\\local_run","700227.json")

model = RlModel(None, False)
with open(args.path, 'r') as f:
    checkpoint_data = json.loads(f.read())
    model.from_packet(checkpoint_data['model'])

# initiate coverage map
start_point = [500.0, 850.0, 32.0]
coverage_map = CoverageMap(start_point=start_point, map_size=12000, scale_ratio=10, state_size=400, input_size=84, height_threshold=0.9, reward_norm=10.0)

print('Connecting to AirSim...')
car_client = airsim.CarClient()
car_client.confirmConnection()
car_client.enableApiControl(True)
car_controls = airsim.CarControls()
coverage_map.set_client(client=car_client)
print('Connected!')

def append_to_ring_buffer(item, buffer, buffer_size):
    if (len(buffer) >= buffer_size):
        buffer = buffer[1:]
    buffer.append(item)
    return buffer

# Gets a coverage image from AirSim
def get_cov_image():

    state, _ = coverage_map.get_state()
    state = state / 255.0

    return state

state_buffer = []
state_buffer_len = 4

print('Running car for a few seconds...')
car_controls.steering = 0
car_controls.throttle = 0.5
car_controls.brake = 0
car_client.setCarControls(car_controls)
stop_run_time = datetime.datetime.now() + datetime.timedelta(seconds=0.1)
while(datetime.datetime.now() < stop_run_time):
    time.sleep(0.01)
    state_buffer = append_to_ring_buffer(get_cov_image(), state_buffer, state_buffer_len)

# slow down a bit
car_controls.throttle = 0.4
car_client.setCarControls(car_controls)

print('Running model')
while(True):
    state_buffer = append_to_ring_buffer(get_cov_image(), state_buffer, state_buffer_len)
    next_state, dummy = model.predict_state(state_buffer)

    # Convert the selected state to a control signal
    next_steering, is_reverse, next_brake = model.state_to_control_signals(next_state, car_client.getCarState())

    # Take the action
    car_controls.steering = next_steering
    car_controls.brake = next_brake
    if is_reverse:
        car_controls.throttle = -0.4
        car_controls.is_manual_gear = True
        car_controls.manual_gear = -1
    else:
        car_controls.throttle = 0.4
        car_controls.is_manual_gear = False
        car_controls.manual_gear = 0

    print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(next_state, car_controls.steering, car_controls.throttle, car_controls.brake))

    car_client.setCarControls(car_controls)

    time.sleep(0.01)